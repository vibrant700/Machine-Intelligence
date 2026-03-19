"""
MNIST 数字识别预测 API - 兼容前端格式
支持画板绘制和图片上传两种方式
"""

import base64
import os
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
# 加载MLP模型
MLP_1_PATH = os.path.join(os.path.dirname(__file__), "MLP_1.pkl")
try:
    with open(MLP_1_PATH, "rb") as f:
        MLP_1 = pickle.load(f)
    MLP_1.eval()
    print("[OK] MLP_1 loaded successfully")
except FileNotFoundError:
    print("[ERROR] MLP_1 model file not found, please train model first")
    MLP_1 = None

# 加载CNN模型
CNN_PATH = os.path.join(os.path.dirname(__file__), "CNN.pkl")
try:
    import torch

    CNN = torch.load(
        CNN_PATH, map_location=torch.device("cpu"), weights_only=False
    )
    CNN.eval()
    print("[OK] CNN loaded successfully")
except FileNotFoundError:
    print("[ERROR] CNN model file not found, please train model first")
    CNN = None

# 默认使用MLP模型
net = MLP_1


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
    预测接口

    接收格式:
    {
        "image_base64": "data:image/png;base64,iVBORw0KGgoAAA...",
        "width": 28,
        "height": 28,
        "normalized_pixels": [0.0, 0.1, ...],  // 可选
        "model_type": "MLP"  // 可选，默认MLP，可选CNN
    }

    返回格式:
    {
        "prediction": 7,
        "confidence": 0.95,
        "probabilities": [0.01, 0.02, ..., 0.90, ...]
    }
    """
    try:
        # 获取请求数据
        data = request.json

        # 选择模型
        model_type = data.get("model_type", "MLP").upper()

        if model_type == "CNN":
            if CNN is None:
                return jsonify({"error": "CNN模型未加载，请先训练模型"}), 500
            selected_model = CNN
            model_name = "CNN"
        else:  # 默认使用MLP
            if MLP_1 is None:
                return jsonify({"error": "MLP模型未加载，请先训练模型"}), 500
            selected_model = MLP_1
            model_name = "MLP"

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

        # 根据模型类型进行不同的预处理
        if model_type == "CNN":
            # CNN使用PyTorch，需要(channel, height, width)格式
            # 注意：训练时用的是 0-255 范围的数据，所以需要转换回来
            import torch

            img_cnn = (img * 255).astype(np.float32)  # 转换回 0-255
            img_tensor = (
                torch.tensor(img_cnn, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            with torch.no_grad():
                logits = selected_model(img_tensor)
            probabilities = (
                torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            )
        else:
            # MLP使用特征提取器
            img_uint8 = (img * 255).astype(np.uint8)
            img_batch = img_uint8.reshape(1, 28, 28)
            features = extract_features_batch(img_batch)
            logits = selected_model(features)

            # 应用 Softmax 将 logits 转换为概率
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
                "distribution": probabilities.tolist(),  # 备用字段
                "scores": probabilities.tolist(),  # 备用字段
                "model_type": model_name,
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
            "mlp_loaded": MLP_1 is not None,
            "cnn_loaded": CNN is not None,
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
                "POST /predict": "主预测接口（兼容前端，支持MLP/CNN）",
                "POST /predict/canvas": "画板预测",
                "POST /predict/image": "图片上传预测",
                "GET /health": "健康检查",
            },
            "models": {
                "MLP": "loaded" if MLP_1 else "not_loaded",
                "CNN": "loaded" if CNN else "not_loaded",
            },
        }
    )


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
