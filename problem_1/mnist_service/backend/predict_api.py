"""
MNIST 数字识别预测 API - 兼容前端格式
支持画板绘制和图片上传两种方式
"""

import base64
import pickle
from io import BytesIO

import cv2
import numpy as np
from feature_extractor import extract_features_batch
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)  # 允许跨域请求

print("Loading model...")
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "try_features1_model.pkl")
try:
    with open(MODEL_PATH, "rb") as f:
        net = pickle.load(f)
    net.eval()
    print("[OK] Model loaded successfully")
except FileNotFoundError:
    print("[ERROR] Model file not found, please train model first")
    net = None


def base64_to_image(base64_string):
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    # Base64解码
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_bytes))
    if image.mode != "L":
        image = image.convert("L")
    # 颜色反转与二值化
    img = np.array(image, dtype=np.uint8)
    if img.mean() > 127:  # 判断背景色
        img = 255 - img  # 反转颜色（黑底白字→白底黑字）
    # 高斯模糊去噪
    # 产生中间值
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # OTSU自适应阈值二值化
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 数字区域检测与裁剪
    ys, xs = np.where(binary > 0)
    if ys.size == 0 or xs.size == 0:
        return np.zeros((28, 28), dtype=np.float32)

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    pad = 2
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(img.shape[0] - 1, y1 + pad)
    x1 = min(img.shape[1] - 1, x1 + pad)

    img = img[y0 : y1 + 1, x0 : x1 + 1]
    binary = binary[y0 : y1 + 1, x0 : x1 + 1]
    # 形态学处理
    foreground_ratio = float(np.count_nonzero(binary)) / float(binary.size)
    if foreground_ratio < 0.03:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        iterations = 2 if foreground_ratio < 0.01 else 1
        binary = cv2.dilate(binary, kernel, iterations=iterations)

    img = cv2.bitwise_and(img, img, mask=(binary > 0).astype(np.uint8))

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((28, 28), dtype=np.float32)
    # 缩放与居中
    scale = 20.0 / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    # 当缩小图像时，多个像素合并成一个像素,合并计算会产生中间值
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    bin_resized = cv2.resize(
        binary, (new_w, new_h), interpolation=cv2.INTER_NEAREST
    )

    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas_bin = np.zeros((28, 28), dtype=np.uint8)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = img_resized
    canvas_bin[top : top + new_h, left : left + new_w] = bin_resized
    # 质心居中
    m = cv2.moments(canvas_bin)
    if m["m00"] != 0:
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        dx = int(round(14 - cx))
        dy = int(round(14 - cy))
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        canvas = cv2.warpAffine(
            canvas, M, (28, 28), flags=cv2.INTER_NEAREST, borderValue=0
        )
    # 归一化后返回结果
    return canvas.astype(np.float32) / 255.0


@app.route("/predict", methods=["POST"])
def predict():
    """
    主预测接口 - 兼容前端格式

    接收格式:
    {
        "image_base64": "data:image/png;base64,iVBORw0KGgoAAA...",
        "width": 28,
        "height": 28,
        "normalized_pixels": [0.0, 0.1, ...]  // 可选
    }

    返回格式:
    {
        "prediction": 7,
        "confidence": 0.95,
        "probabilities": [0.01, 0.02, ..., 0.90, ...]
    }
    """
    try:
        if net is None:
            return jsonify({"error": "模型未加载，请先训练模型"}), 500

        # 获取请求数据
        data = request.json

        # 方法 1: 从 image_base64 提取
        if "image_base64" in data:
            img = base64_to_image(data["image_base64"])

        # 方法 2: 从 normalized_pixels 提取
        elif "normalized_pixels" in data:
            pixels = np.array(data["normalized_pixels"], dtype=np.float32)
            width = data.get("width", 28)
            height = data.get("height", 28)
            img = pixels.reshape(height, width)

        else:
            return jsonify({"error": "缺少图片数据"}), 400

        # 确保是 28x28
        if img.shape != (28, 28):
            img = cv2.resize(img, (28, 28))

        # 提取特征
        features = extract_features_batch(img[np.newaxis, ...])

        # 预测
        logits = net(features)

        # 应用 Softmax 将 logits 转换为概率
        def softmax(x):
            exp_x = np.exp(x - np.max(x))  # 数值稳定
            return exp_x / exp_x.sum()

        probabilities = softmax(logits[0])
        predicted_label = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_label])

        return jsonify(
            {
                "prediction": predicted_label,
                "confidence": confidence,
                "probabilities": probabilities.tolist(),
                "distribution": probabilities.tolist(),  # 备用字段
                "scores": probabilities.tolist(),  # 备用字段
            }
        )

    except Exception as e:
        import traceback

        return (
            jsonify({"error": str(e), "traceback": traceback.format_exc()}),
            500,
        )


@app.route("/predict/canvas", methods=["POST"])
def predict_canvas():
    """
    画板预测接口（备用）

    接收格式:
    {
        "canvas": {
            "width": 280,
            "height": 280,
            "pixels": [[255,255,255,255], ...]
        }
    }
    """
    try:
        if net is None:
            return jsonify({"error": "模型未加载"}), 500

        data = request.json
        canvas_data = data.get("canvas")

        if not canvas_data:
            return jsonify({"error": "缺少画板数据"}), 400

        width = canvas_data["width"]
        height = canvas_data["height"]
        pixels = np.array(canvas_data["pixels"], dtype=np.uint8)

        # 提取 RGB 通道（忽略 Alpha）
        if len(pixels.shape) == 3 and pixels.shape[-1] == 4:
            pixels = pixels[:, :, :3]

        # 转换为灰度图
        if len(pixels.shape) == 3:
            gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        else:
            gray = pixels

        # 反转颜色（黑底白字 -> 白底黑字）
        gray = 255 - gray

        # 调整大小为 28x28
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

        # 归一化
        img = resized.astype(np.float32) / 255.0

        # 提取特征并预测
        features = extract_features_batch(img[np.newaxis, ...])
        logits = net(features)

        # 应用 Softmax
        def softmax(x):
            exp_x = np.exp(x - np.max(x))
            return exp_x / exp_x.sum()

        probabilities = softmax(logits[0])
        predicted_label = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_label])

        return jsonify(
            {
                "prediction": predicted_label,
                "confidence": confidence,
                "probabilities": probabilities.tolist(),
            }
        )

    except Exception as e:
        import traceback

        return (
            jsonify({"error": str(e), "traceback": traceback.format_exc()}),
            500,
        )


@app.route("/predict/image", methods=["POST"])
def predict_image():
    """
    图片上传预测接口（备用）

    接收格式:
    {
        "image": "data:image/png;base64,iVBORw0KGgoAAA..."
    }
    """
    try:
        if net is None:
            return jsonify({"error": "模型未加载"}), 500

        data = request.json
        image_data = data.get("image")

        if not image_data:
            return jsonify({"error": "缺少图片数据"}), 400

        # 转换图片
        img = base64_to_image(image_data)

        # 提取特征并预测
        features = extract_features_batch(img[np.newaxis, ...])
        logits = net(features)

        # 应用 Softmax
        def softmax(x):
            exp_x = np.exp(x - np.max(x))
            return exp_x / exp_x.sum()

        probabilities = softmax(logits[0])
        predicted_label = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_label])

        return jsonify(
            {
                "prediction": predicted_label,
                "confidence": confidence,
                "probabilities": probabilities.tolist(),
            }
        )

    except Exception as e:
        import traceback

        return (
            jsonify({"error": str(e), "traceback": traceback.format_exc()}),
            500,
        )


@app.route("/health", methods=["GET"])
def health():
    """健康检查"""
    return jsonify(
        {
            "status": "ok",
            "model_loaded": net is not None,
            "message": "MNIST 预测 API 运行中",
        }
    )


@app.route("/", methods=["GET"])
def index():
    """API 信息"""
    return jsonify(
        {
            "service": "MNIST Digit Recognition API",
            "version": "1.0",
            "endpoints": {
                "POST /predict": "主预测接口（兼容前端）",
                "POST /predict/canvas": "画板预测",
                "POST /predict/image": "图片上传预测",
                "GET /health": "健康检查",
            },
            "model_status": "loaded" if net else "not_loaded",
        }
    )


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("[START] MNIST 数字识别 API 服务启动")
    print("=" * 70)
    print("[ADDR] 地址: http://localhost:5000")
    print("[ADDR] 本地: http://127.0.0.1:5000")
    print("")
    print("[INFO] 可用接口:")
    print("   POST /predict         - 主预测接口（前端默认）")
    print("   POST /predict/canvas  - 画板预测")
    print("   POST /predict/image  - 图片上传预测")
    print("   GET  /health          - 健康检查")
    print("   GET  /                - API 信息")
    print("=" * 70)
    print("[WAIT] 等待前端连接...")
    print("=" * 70 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
