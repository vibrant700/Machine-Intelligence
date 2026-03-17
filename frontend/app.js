const drawCanvas = document.getElementById("drawCanvas");
const drawCtx = drawCanvas.getContext("2d");
const brushSizeInput = document.getElementById("brushSize");
const brushSizeValue = document.getElementById("brushSizeValue");
const brushColorInput = document.getElementById("brushColor");
const undoBtn = document.getElementById("undoBtn");
const clearBtn = document.getElementById("clearBtn");
const fileInput = document.getElementById("fileInput");
const predictBtn = document.getElementById("predictBtn");
const cancelBtn = document.getElementById("cancelBtn");
const apiEndpointInput = document.getElementById("apiEndpoint");
const statusText = document.getElementById("statusText");
const predictedDigitEl = document.getElementById("predictedDigit");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceText = document.getElementById("confidenceText");
const timeText = document.getElementById("timeText");
const preprocessTimeText = document.getElementById("preprocessTimeText");
const apiTimeText = document.getElementById("apiTimeText");
const processedCanvas = document.getElementById("processedCanvas");
const processedCtx = processedCanvas.getContext("2d");
const heatmapCanvas = document.getElementById("heatmapCanvas");
const heatmapCtx = heatmapCanvas.getContext("2d");
const historyList = document.getElementById("historyList");
const historyTemplate = document.getElementById("historyItemTemplate");

const preprocessCache = new Map();
const predictionCache = new Map();
const activeRequests = new Map();
const drawHistory = [];
const resultHistory = [];
const MAX_HISTORY = 20;
const MAX_DRAW_HISTORY = 30;
const DEFAULT_API_ENDPOINT = "http://localhost:8000/predict";

let isDrawing = false;
let lastPoint = null;
let hasContent = false;

initCanvas();
initEvents();

function initCanvas() {
  drawCtx.fillStyle = "#ffffff";
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  drawCtx.lineCap = "round";
  drawCtx.lineJoin = "round";
  drawCtx.strokeStyle = brushColorInput.value;
  drawCtx.lineWidth = Number(brushSizeInput.value);
}

function initEvents() {
  drawCanvas.addEventListener("pointerdown", startDrawing);
  drawCanvas.addEventListener("pointermove", draw);
  drawCanvas.addEventListener("pointerup", stopDrawing);
  drawCanvas.addEventListener("pointerleave", stopDrawing);
  drawCanvas.addEventListener("pointercancel", stopDrawing);

  brushSizeInput.addEventListener("input", () => {
    drawCtx.lineWidth = Number(brushSizeInput.value);
    brushSizeValue.textContent = `${brushSizeInput.value}px`;
  });

  brushColorInput.addEventListener("input", () => {
    drawCtx.strokeStyle = brushColorInput.value;
  });

  clearBtn.addEventListener("click", clearCanvas);
  undoBtn.addEventListener("click", undoCanvas);
  fileInput.addEventListener("change", handleFileUpload);
  predictBtn.addEventListener("click", handlePredict);
  cancelBtn.addEventListener("click", cancelAllRequests);

  window.addEventListener("error", (event) => {
    setStatus(`发生错误：${event.message}`, true);
    // 调试日志（定位全局运行时错误，稳定后可删除）
    console.error("GlobalError", event.error || event.message);
  });

  window.addEventListener("unhandledrejection", (event) => {
    setStatus("发生异步错误，请检查接口与网络。", true);
    // 调试日志（定位未捕获 Promise 异常，稳定后可删除）
    console.error("UnhandledRejection", event.reason);
  });
}

function startDrawing(event) {
  drawCanvas.setPointerCapture(event.pointerId);
  saveHistorySnapshot();
  isDrawing = true;
  lastPoint = getPoint(event);
}

function draw(event) {
  if (!isDrawing) {
    return;
  }
  const point = getPoint(event);
  drawCtx.beginPath();
  drawCtx.moveTo(lastPoint.x, lastPoint.y);
  drawCtx.lineTo(point.x, point.y);
  drawCtx.stroke();
  lastPoint = point;
  hasContent = true;
}

function stopDrawing(event) {
  if (!isDrawing) {
    return;
  }
  isDrawing = false;
  drawCanvas.releasePointerCapture(event.pointerId);
}

function getPoint(event) {
  const rect = drawCanvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * (drawCanvas.width / rect.width),
    y: (event.clientY - rect.top) * (drawCanvas.height / rect.height),
  };
}

function saveHistorySnapshot() {
  if (drawHistory.length >= MAX_DRAW_HISTORY) {
    drawHistory.shift();
  }
  drawHistory.push(drawCanvas.toDataURL("image/png"));
}

function clearCanvas() {
  saveHistorySnapshot();
  drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
  drawCtx.fillStyle = "#ffffff";
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  hasContent = false;
  setStatus("已清除画布");
}

function undoCanvas() {
  const last = drawHistory.pop();
  if (!last) {
    setStatus("没有可撤销的操作");
    return;
  }
  const img = new Image();
  img.onload = () => {
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    drawCtx.drawImage(img, 0, 0);
    hasContent = true;
    setStatus("已撤销");
  };
  img.src = last;
}

async function handleFileUpload(event) {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }
  const validType = ["image/png", "image/jpeg", "image/gif"].includes(file.type);
  if (!validType) {
    setStatus("仅支持 PNG/JPG/GIF 图片", true);
    return;
  }
  saveHistorySnapshot();
  const fileDataUrl = await fileToDataUrl(file);
  const image = await loadImage(fileDataUrl);
  drawCtx.fillStyle = "#ffffff";
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  drawImageContain(drawCtx, image, drawCanvas.width, drawCanvas.height);
  hasContent = true;
  setStatus("图片已导入");
}

async function handlePredict() {
  if (!hasContent && isCanvasBlank(drawCtx, drawCanvas)) {
    setStatus("请先在画板输入数字或上传图片", true);
    return;
  }

  const requestId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const endpoint = (apiEndpointInput?.value || DEFAULT_API_ENDPOINT).trim();
  if (!endpoint) {
    setStatus("请填写 API 地址", true);
    return;
  }

  setStatus("正在预处理图像...", false, true);
  const preprocessStart = performance.now();
  const sourceDataUrl = drawCanvas.toDataURL("image/png");
  const sourceHash = hashString(sourceDataUrl);

  let preprocessResult;
  const cachedPreprocess = preprocessCache.get(sourceHash);
  if (cachedPreprocess) {
    preprocessResult = cachedPreprocess;
  } else {
    preprocessResult = preprocessImage(drawCanvas, { applyEnhancement: true });
    preprocessCache.set(sourceHash, preprocessResult);
  }

  const preprocessCost = performance.now() - preprocessStart;
  preprocessTimeText.textContent = `${preprocessCost.toFixed(2)} ms`;
  if (preprocessCost > 500) {
    setStatus(`预处理耗时 ${preprocessCost.toFixed(0)}ms，已超过 500ms 目标`, true);
  }

  renderProcessedPreview(preprocessResult.grayscaleImageData, preprocessResult.normalizedPixels);

  const payload = {
    image_base64: preprocessResult.base64,
    image_binary: Array.from(preprocessResult.binary),
    width: 28,
    height: 28,
    normalized_pixels: preprocessResult.normalizedPixels,
  };

  const responseCacheKey = `${sourceHash}:${endpoint}`;
  const cachedPrediction = predictionCache.get(responseCacheKey);
  if (cachedPrediction) {
    renderPrediction(cachedPrediction, preprocessCost, 0, true);
    appendHistory(cachedPrediction, preprocessCost, 0, true);
    setStatus("命中缓存，已返回历史识别结果");
    return;
  }

  const controller = new AbortController();
  activeRequests.set(requestId, controller);

  try {
    setStatus("正在调用识别接口...", false, true);
    const apiStart = performance.now();
    const apiData = await requestPrediction(endpoint, payload, {
      signal: controller.signal,
      timeoutMs: 30000,
      retries: 2,
    });
    const apiCost = performance.now() - apiStart;
    const parsed = normalizeApiResponse(apiData);
    predictionCache.set(responseCacheKey, parsed);
    renderPrediction(parsed, preprocessCost, apiCost, false);
    appendHistory(parsed, preprocessCost, apiCost, false);
    if (apiCost > 3000) {
      setStatus(`请求耗时 ${apiCost.toFixed(0)}ms，超过 3 秒目标`, true);
    } else {
      setStatus("识别完成");
    }
  } catch (error) {
    if (error.name === "AbortError") {
      setStatus("请求已取消", true);
    } else {
      setStatus(`识别失败：${error.message}`, true);
      // 调试日志（接口联调阶段用于查看完整报错堆栈，稳定后可删除）
      console.error("PredictError", error);
    }
  } finally {
    activeRequests.delete(requestId);
  }
}

function cancelAllRequests() {
  if (!activeRequests.size) {
    setStatus("当前无进行中的请求");
    return;
  }
  activeRequests.forEach((controller) => controller.abort());
  activeRequests.clear();
  setStatus("已取消全部请求", true);
}

function preprocessImage(sourceCanvas, options = { applyEnhancement: true }) {
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = sourceCanvas.width;
  tempCanvas.height = sourceCanvas.height;
  const tempCtx = tempCanvas.getContext("2d", { willReadFrequently: true });
  tempCtx.drawImage(sourceCanvas, 0, 0);
  const source = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
  const intensity = new Float32Array(tempCanvas.width * tempCanvas.height);

  for (let i = 0; i < source.data.length; i += 4) {
    const r = source.data[i];
    const g = source.data[i + 1];
    const b = source.data[i + 2];
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;
    const inv = 255 - gray;
    intensity[i / 4] = inv < 24 ? 0 : inv;
  }

  const bbox = findBoundingBox(intensity, tempCanvas.width, tempCanvas.height);
  if (!bbox) {
    throw new Error("未检测到有效笔迹");
  }

  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = 28;
  cropCanvas.height = 28;
  const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });
  cropCtx.fillStyle = "#000000";
  cropCtx.fillRect(0, 0, 28, 28);

  const pad = Math.max(Math.round(Math.max(bbox.w, bbox.h) * 0.2), 2);
  const sx = Math.max(0, bbox.x - pad);
  const sy = Math.max(0, bbox.y - pad);
  const sw = Math.min(tempCanvas.width - sx, bbox.w + pad * 2);
  const sh = Math.min(tempCanvas.height - sy, bbox.h + pad * 2);

  const normalizedCanvas = document.createElement("canvas");
  normalizedCanvas.width = sw;
  normalizedCanvas.height = sh;
  const normalizedCtx = normalizedCanvas.getContext("2d");
  const patch = imageDataFromIntensity(intensity, tempCanvas.width, tempCanvas.height);
  tempCtx.putImageData(patch, 0, 0);
  normalizedCtx.drawImage(tempCanvas, sx, sy, sw, sh, 0, 0, sw, sh);

  const fit = fitContain(sw, sh, 20, 20);
  const offsetX = Math.floor((28 - fit.w) / 2);
  const offsetY = Math.floor((28 - fit.h) / 2);
  cropCtx.drawImage(normalizedCanvas, 0, 0, sw, sh, offsetX, offsetY, fit.w, fit.h);

  let imageData = cropCtx.getImageData(0, 0, 28, 28);
  const gray = new Float32Array(28 * 28);
  for (let i = 0; i < imageData.data.length; i += 4) {
    gray[i / 4] = imageData.data[i];
  }

  let enhanced = gray;
  if (options.applyEnhancement) {
    enhanced = edgeEnhance(morphClose(gray, 28, 28), 28, 28);
  }

  const finalImageData = new ImageData(28, 28);
  const normalizedPixels = [];
  const binary = new Uint8Array(28 * 28);
  for (let i = 0; i < enhanced.length; i += 1) {
    const value = Math.max(0, Math.min(255, enhanced[i]));
    finalImageData.data[i * 4] = value;
    finalImageData.data[i * 4 + 1] = value;
    finalImageData.data[i * 4 + 2] = value;
    finalImageData.data[i * 4 + 3] = 255;
    const normalized = Number((value / 255).toFixed(6));
    normalizedPixels.push(normalized);
    binary[i] = Math.round(normalized * 255);
  }

  cropCtx.putImageData(finalImageData, 0, 0);

  return {
    grayscaleImageData: finalImageData,
    normalizedPixels,
    binary,
    base64: cropCanvas.toDataURL("image/png"),
  };
}

function renderProcessedPreview(imageData, normalizedPixels) {
  processedCtx.putImageData(imageData, 0, 0);
  const heatmapData = new ImageData(28, 28);
  for (let i = 0; i < normalizedPixels.length; i += 1) {
    const v = normalizedPixels[i];
    const r = Math.round(255 * v);
    const g = Math.round(90 * (1 - v));
    const b = Math.round(255 * (1 - v));
    heatmapData.data[i * 4] = r;
    heatmapData.data[i * 4 + 1] = g;
    heatmapData.data[i * 4 + 2] = b;
    heatmapData.data[i * 4 + 3] = 255;
  }
  heatmapCtx.putImageData(heatmapData, 0, 0);
}

async function requestPrediction(endpoint, payload, options) {
  const { signal, timeoutMs, retries } = options;
  let attempts = 0;
  let lastError = null;

  while (attempts <= retries) {
    const timeoutController = new AbortController();
    const timeoutId = setTimeout(() => timeoutController.abort(), timeoutMs);
    const linkedSignal = anySignal([signal, timeoutController.signal]);

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify(payload),
        signal: linkedSignal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      clearTimeout(timeoutId);
      return data;
    } catch (error) {
      clearTimeout(timeoutId);
      if (error.name === "AbortError" && signal?.aborted) {
        throw error;
      }
      lastError = error;
      attempts += 1;
      if (attempts > retries) {
        break;
      }
      await delay(350 * attempts);
    }
  }

  throw new Error(`请求失败，已重试 ${retries} 次：${lastError?.message || "未知错误"}`);
}

function normalizeApiResponse(data) {
  const probabilitySource = Array.isArray(data.probabilities)
    ? data.probabilities
    : Array.isArray(data.distribution)
      ? data.distribution
      : Array.isArray(data.scores)
        ? data.scores
        : [];
  const probabilities = normalizeProbabilities(probabilitySource);

  const predictedDigit =
    Number.isInteger(data.prediction) && data.prediction >= 0 && data.prediction <= 9
      ? data.prediction
      : probabilities.indexOf(Math.max(...probabilities));

  const confidence =
    typeof data.confidence === "number"
      ? clamp(data.confidence, 0, 1)
      : clamp(probabilities[predictedDigit] || 0, 0, 1);

  return {
    predictedDigit,
    confidence,
    probabilities,
  };
}

function renderPrediction(result, preprocessCost, apiCost, fromCache) {
  const total = preprocessCost + apiCost;
  predictedDigitEl.textContent = String(result.predictedDigit);
  const percent = result.confidence * 100;
  confidenceBar.style.width = `${percent.toFixed(2)}%`;
  confidenceText.textContent = `${percent.toFixed(2)}%${fromCache ? "（缓存）" : ""}`;
  timeText.textContent = `${total.toFixed(2)} ms`;
  apiTimeText.textContent = `${apiCost.toFixed(2)} ms`;
}

function appendHistory(result, preprocessCost, apiCost, fromCache) {
  const item = {
    predictedDigit: result.predictedDigit,
    confidence: result.confidence,
    totalMs: preprocessCost + apiCost,
    createdAt: new Date(),
    fromCache,
  };
  resultHistory.unshift(item);
  if (resultHistory.length > MAX_HISTORY) {
    resultHistory.pop();
  }
  renderHistory();
}

function renderHistory() {
  historyList.innerHTML = "";
  const frag = document.createDocumentFragment();
  resultHistory.forEach((item) => {
    const node = historyTemplate.content.cloneNode(true);
    node.querySelector(".history-digit").textContent = `数字 ${item.predictedDigit}`;
    node.querySelector(".history-time").textContent = item.createdAt.toLocaleTimeString();
    node.querySelector(".history-sub").textContent = `置信度 ${(item.confidence * 100).toFixed(2)}% | 总耗时 ${item.totalMs.toFixed(2)}ms${item.fromCache ? " | 缓存" : ""}`;
    frag.appendChild(node);
  });
  historyList.appendChild(frag);
}

function setStatus(text, isError = false, isLoading = false) {
  statusText.textContent = text;
  statusText.classList.toggle("error", isError);
  statusText.classList.toggle("loading", isLoading);
}

function findBoundingBox(intensity, width, height) {
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const value = intensity[y * width + x];
      if (value > 10) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < minX || maxY < minY) {
    return null;
  }
  return {
    x: minX,
    y: minY,
    w: maxX - minX + 1,
    h: maxY - minY + 1,
  };
}

function imageDataFromIntensity(intensity, width, height) {
  const output = new ImageData(width, height);
  for (let i = 0; i < intensity.length; i += 1) {
    const value = Math.max(0, Math.min(255, intensity[i]));
    output.data[i * 4] = value;
    output.data[i * 4 + 1] = value;
    output.data[i * 4 + 2] = value;
    output.data[i * 4 + 3] = 255;
  }
  return output;
}

function morphClose(gray, width, height) {
  return erode(dilate(gray, width, height), width, height);
}

function dilate(gray, width, height) {
  const out = new Float32Array(gray.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let max = 0;
      for (let ky = -1; ky <= 1; ky += 1) {
        for (let kx = -1; kx <= 1; kx += 1) {
          const nx = x + kx;
          const ny = y + ky;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
            continue;
          }
          const v = gray[ny * width + nx];
          if (v > max) {
            max = v;
          }
        }
      }
      out[y * width + x] = max;
    }
  }
  return out;
}

function erode(gray, width, height) {
  const out = new Float32Array(gray.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let min = 255;
      for (let ky = -1; ky <= 1; ky += 1) {
        for (let kx = -1; kx <= 1; kx += 1) {
          const nx = x + kx;
          const ny = y + ky;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
            continue;
          }
          const v = gray[ny * width + nx];
          if (v < min) {
            min = v;
          }
        }
      }
      out[y * width + x] = min;
    }
  }
  return out;
}

function edgeEnhance(gray, width, height) {
  const out = new Float32Array(gray.length);
  const kernelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const kernelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      let gx = 0;
      let gy = 0;
      let idx = 0;
      for (let ky = -1; ky <= 1; ky += 1) {
        for (let kx = -1; kx <= 1; kx += 1) {
          const value = gray[(y + ky) * width + (x + kx)];
          gx += value * kernelX[idx];
          gy += value * kernelY[idx];
          idx += 1;
        }
      }
      const edge = Math.sqrt(gx * gx + gy * gy);
      const base = gray[y * width + x];
      out[y * width + x] = clamp(base * 0.78 + edge * 0.22, 0, 255);
    }
  }
  return out;
}

function fitContain(sourceW, sourceH, maxW, maxH) {
  const scale = Math.min(maxW / sourceW, maxH / sourceH);
  return {
    w: Math.max(1, Math.floor(sourceW * scale)),
    h: Math.max(1, Math.floor(sourceH * scale)),
  };
}

function drawImageContain(ctx, image, w, h) {
  const fit = fitContain(image.width, image.height, w, h);
  const x = Math.floor((w - fit.w) / 2);
  const y = Math.floor((h - fit.h) / 2);
  ctx.drawImage(image, 0, 0, image.width, image.height, x, y, fit.w, fit.h);
}

function normalizeProbabilities(input) {
  if (!Array.isArray(input) || input.length !== 10) {
    return Array(10).fill(0.1);
  }
  const cleaned = input.map((v) => (Number.isFinite(v) ? Math.max(v, 0) : 0));
  const sum = cleaned.reduce((acc, v) => acc + v, 0);
  if (sum <= 0) {
    return Array(10).fill(0.1);
  }
  return cleaned.map((v) => v / sum);
}

function anySignal(signals) {
  const controller = new AbortController();
  const onAbort = () => controller.abort();
  signals.filter(Boolean).forEach((signal) => {
    if (signal.aborted) {
      controller.abort();
    } else {
      signal.addEventListener("abort", onAbort, { once: true });
    }
  });
  return controller.signal;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("文件读取失败"));
    reader.readAsDataURL(file);
  });
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("图片加载失败"));
    img.src = src;
  });
}

function isCanvasBlank(ctx, canvas) {
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  for (let i = 0; i < imageData.length; i += 4) {
    if (imageData[i] !== 255 || imageData[i + 1] !== 255 || imageData[i + 2] !== 255) {
      return false;
    }
  }
  return true;
}

function clamp(v, min, max) {
  return Math.min(max, Math.max(min, v));
}

function hashString(str) {
  let h = 5381;
  for (let i = 0; i < str.length; i += 1) {
    h = (h << 5) + h + str.charCodeAt(i);
    h |= 0;
  }
  return String(h);
}
