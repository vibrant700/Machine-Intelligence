(function () {
  'use strict';

  if (window.adSkipper) return;

  // Sidebar state (will be initialized when sidebar loads)
  let sidebarState = null;

  // API 基础路径
  const VIDEO_ANALYSIS_BASE = window.LOCAL_CONFIG?.API_BASE || 'http://localhost:8080';
  const DANMU_TRIGGER_WINDOW_SEC = 0.3;
  const DANMU_REWIND_RESET_SEC = 1.0;
  const DANMU_MAX_CONCURRENT = 3;
  const DANMU_BASE_DURATION_SEC = 7.5;
  const DANMU_FALLBACK_TRACK_PX = 640;
  const DANMU_NATIVE_SAMPLE_MS = 300;
  const DANMU_SPEED_MIN_PX_PER_SEC = 60;
  const DANMU_SPEED_MAX_PX_PER_SEC = 1200;
  const DANMU_SPEED_TUNE_FACTOR = 0.78;
  const DANMU_REMOVE_BUFFER_PX = 32;
  const DANMU_LANE_MIN_PERCENT = 6;
  const DANMU_LANE_MAX_PERCENT = 25;
  const DANMU_LANE_STEP_PERCENT = 4;
  const DANMU_DEFAULT_LANES = [8, 14, 20, 24];
  const DANMU_NATIVE_NODE_SELECTORS = [
    '.bpx-player-dm-wrap .bili-dm',
    '.bpx-player-dm-wrap [class*="dm"][class*="item"]',
    '.bilibili-player-video-danmaku [class*="danmaku"]'
  ];

  class AdSkipperCore {
    constructor() {
      this.player = new BilibiliPlayerController();
      this.sidebarController = null;
      this.segments = [];
      this.allSegments = [];
      this.aiSummary = '';
      this.lastSkipTime = 0;
      // 日志控制变量
      this.noSegmentLogPrinted = false;
      this.coolDownLogPrinted = false;
      this.matchProcessLogPrinted = false;
      this.noAdMatchLogPrinted = false;

      this.networkState = {
        offlineUntil: 0,
        hasLoggedOffline: false,
        wasOffline: false
      };
      this.networkCooldownMs = 30000;
      this.networkTimeoutMs = 6000;
      this.analysisBvid = null;
      this.knowledgeDanmuQueue = [];
      this.triggeredDanmuIds = new Set();
      this.knowledgeDanmuLayer = null;
      this.lastDanmuCurrentTime = 0;
      this.nextDanmuLane = 0;
      this.activeKnowledgeDanmus = [];
      this.knowledgeDanmuRafId = null;
      this.lastDanmuFrameTs = 0;
      this.nativeDanmuSpeedPxPerSec = 0;
      this.lastNativeSpeedSampleTs = 0;
      this.nativeDanmuTrackSample = null;


    }

    init() {
      console.log("[AdSkipper] 初始化...");
      this.player.init().then(ok => {
        if (!ok) return;

        // 检查登录状态
        chrome.storage.local.get(['adskipper_token'], (storage) => {
          const token = storage.adskipper_token;
          console.log('[AdSkipper] 登录状态:', token ? '已登录' : '未登录');
        });

        // ==========================
        // Vue 侧边栏初始化
        // ==========================
        this.initSidebar().then(() => {
          console.log('[AdSkipper] 侧边栏初始化完成');
        }).catch(err => {
          console.error("[AdSkipper] Sidebar 初始化失败:", err);
        });

        this.initAiFloatingButton();

        this.player.onTimeUpdate = (t) => this.checkSkip(t);

        const bvid = this.player.currentBvid;
        if (bvid) {
          this.refreshAnalysisForBvid(bvid).then(() => {
            window.adSkipper = this;
          });
        }
      });

      // ESC key listener
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
          if (this.sidebarController) {
            this.sidebarController.hide();
          }
        }
      });

      window.addEventListener('visionmark:seek', (event) => {
        const time = Number(event?.detail?.time);
        if (!Number.isFinite(time)) return;
        this.player.skipTo(Math.max(time, 0));
      });

      window.addEventListener('visionmark:refresh-ai', () => {
        if (this.player.currentBvid) {
          this.refreshAnalysisForBvid(this.player.currentBvid, { forceAnalyze: true });
        }
      });

      // 调试：将实例暴露到全局，便于控制台调试
      window.adSkipperDebug = this;
      console.log('[AdSkipper] 调试模式已启用，使用: adSkipperDebug.addSegmentMarkers() 手动添加标记');
    }

    async initSidebar() {
      if (this.sidebarController) return;

      const { createSidebar, sidebarState: importedSidebarState } = await import('../sidebar/index.js');
      sidebarState = importedSidebarState;

      const existingRoot = document.getElementById('vm-sidebar-root');
      if (existingRoot) {
        this.sidebarController = createSidebar(existingRoot);
        return;
      }

      const root = document.createElement('div');
      root.id = 'vm-sidebar-root';
      document.body.appendChild(root);

      console.log('[AdSkipper Sidebar] 正在初始化侧边栏...');
      this.sidebarController = createSidebar(root);
    }

    initAiFloatingButton() {
      console.log("[AdSkipper] 创建AI浮动按钮...");

      if (!document.getElementById('visionmark-ai-fab-style')) {
        console.log("[AdSkipper] 添加AI按钮样式...");
        const style = document.createElement('style');
        style.id = 'visionmark-ai-fab-style';
        style.textContent = `
          #visionmark-ai-fab {
            position: fixed !important;
            left: 20px !important;
            top: 100px !important;
            z-index: 2147483647 !important;
            min-width: 46px;
            height: 46px;
            border: none;
            border-radius: 999px;
            background: linear-gradient(135deg, #FB7299, #ff8e9f);
            color: #fff;
            font-size: 14px;
            font-weight: 700;
            letter-spacing: 0.4px;
            cursor: pointer;
            box-shadow: 0 10px 28px rgba(251, 114, 153, 0.45);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
            padding: 0 16px;
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
          }
          #visionmark-ai-fab:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 34px rgba(251, 114, 153, 0.55);
          }
        `;
        document.head.appendChild(style);
        console.log('[AdSkipper] AI button style injected.');
      }

      if (document.getElementById('visionmark-ai-fab')) {
        console.log("[AdSkipper] AI按钮已存在，跳过创建");
        return;
      }

      console.log("[AdSkipper] 创建AI按钮DOM元素...");
      const button = document.createElement('button');
      button.id = 'visionmark-ai-fab';
      button.type = 'button';
      button.textContent = 'AI';
      button.title = '视频总结';
      button.setAttribute('aria-label', '视频总结');
      button.onclick = (event) => {
        event.stopPropagation();
        this.toggleSidebar().catch((error) => {
          console.error('[AdSkipper] 切换侧边栏失败:', error);
          this.showToast('视频总结面板加载失败', 'error');
        });
      };

      document.body.appendChild(button);
      console.log("[AdSkipper] AI按钮已创建并添加到body");

      // 验证按钮是否真的在DOM中并检查位置
      setTimeout(() => {
        const btn = document.getElementById('visionmark-ai-fab');
        if (btn) {
          const rect = btn.getBoundingClientRect();
          console.log('[AdSkipper] AI button is present in DOM.');
          console.log("[AdSkipper] 位置信息:");
          console.log("  - 灏哄:", btn.offsetWidth, "x", btn.offsetHeight);
          console.log("  - 灞忓箷浣嶇疆:", rect.left, ",", rect.top);
          console.log("  - right/top:", rect.right, ",", rect.bottom);
          console.log("  - 在视口内:", rect.top >= 0 && rect.left >= 0 && rect.bottom <= window.innerHeight && rect.right <= window.innerWidth);
          console.log("  - z-index:", window.getComputedStyle(btn).zIndex);
        } else {
          console.error("[AdSkipper] 错误：AI按钮创建后未在DOM中找到！");
        }
      }, 100);
    }

    async ensureSidebarReady() {
      if (this.sidebarController) return true;
      try {
        await this.initSidebar();
      } catch (error) {
        console.error('[AdSkipper] ensureSidebarReady failed:', error);
      }
      return Boolean(this.sidebarController);
    }

    async refreshAnalysisForBvid(bvid, options = {}) {
      if (!bvid) return;
      const shouldAnalyze = Boolean(options.forceAnalyze) || this.segments.length === 0 || !this.aiSummary;
      if (shouldAnalyze) {
        await this.analyzeVideo(bvid);
      }
    }

    async showSidebar(options = {}) {
      if (!await this.ensureSidebarReady()) return;

      if (options.refresh && this.player.currentBvid) {
        await this.refreshAnalysisForBvid(this.player.currentBvid, { forceAnalyze: true });
      }

      this.sidebarController.show();
    }

    async toggleSidebar() {
      if (!await this.ensureSidebarReady()) return;
      this.sidebarController.toggle();
    }

    async getToken() {
      return new Promise((resolve) => {
        chrome.storage.local.get(['adskipper_token'], (storage) => {
          resolve(storage.adskipper_token);
        });
      });
    }

    createNetworkUnavailableError() {
      const error = new Error('网络不可用，请稍后重试');
      error.code = 'NETWORK_UNAVAILABLE';
      return error;
    }

    isNetworkFailure(error) {
      if (!error) return false;
      if (error.name === 'AbortError') return true;
      const message = String(error.message || '').toLowerCase();
      if (message.includes('failed to fetch') || message.includes('networkerror') || message.includes('load failed')) {
        return true;
      }
      return error instanceof TypeError;
    }

    markNetworkOffline(context, error) {
      this.networkState.offlineUntil = Date.now() + this.networkCooldownMs;
      this.networkState.wasOffline = true;
      if (!this.networkState.hasLoggedOffline) {
        console.warn(`[AdSkipper] 后端在 ${context} 时不可达，暂停请求 30 秒。`, error);
        this.networkState.hasLoggedOffline = true;
      }
    }

    markNetworkOnline() {
      if (this.networkState.wasOffline) {
        console.info('[AdSkipper] 后端连接已恢复');
      }
      this.networkState.offlineUntil = 0;
      this.networkState.hasLoggedOffline = false;
      this.networkState.wasOffline = false;
    }

    async safeFetch(url, options = {}, context = 'request') {
      const now = Date.now();

      // 检查网络冷却状态
      if (now < this.networkState.offlineUntil) {
        const remainingSec = Math.ceil((this.networkState.offlineUntil - now) / 1000);
        console.warn(`[AdSkipper] safeFetch: 网络冷却中，还需等待 ${remainingSec} 秒`);
        throw this.createNetworkUnavailableError();
      }

      console.log(`[AdSkipper] safeFetch: 开始请求 ${context}`);
      console.log(`[AdSkipper] safeFetch: URL = ${url}`);
      console.log(`[AdSkipper] safeFetch: 超时设置 = ${this.networkTimeoutMs}ms`);

      const controller = options.signal ? null : new AbortController();
      const timeoutId = setTimeout(() => {
        if (controller) {
          console.warn(`[AdSkipper] safeFetch: 请求超时 (${this.networkTimeoutMs}ms)`);
          controller.abort();
        }
      }, this.networkTimeoutMs);

      try {
        const response = await fetch(url, {
          ...options,
          signal: options.signal || (controller ? controller.signal : undefined)
        });
        clearTimeout(timeoutId);
        this.markNetworkOnline();
        console.log(`[AdSkipper] safeFetch: 请求成功，状态 = ${response.status}`);
        return response;
      } catch (error) {
        clearTimeout(timeoutId);
        console.error(`[AdSkipper] safeFetch: 请求失败`, error);
        if (this.isNetworkFailure(error)) {
          this.markNetworkOffline(context, error);
          throw this.createNetworkUnavailableError();
        }
        throw error;
      }
    }

    normalizeSegment(segment, index) {
      const start = Number(segment.start ?? segment.start_time ?? 0);
      const end = Number(segment.end ?? segment.end_time ?? 0);
      const candidateAction = typeof segment.action === 'string' ? segment.action.toLowerCase() : '';
      const action = candidateAction === 'popup' || candidateAction === 'skip' ? candidateAction : 'skip';

      const rawContent = typeof segment.content === 'string' ? segment.content.trim() : null;
      const content = action === 'popup' ? (rawContent || null) : null;

      return {
        ...segment,
        id: segment.id ?? `${start}-${end}-${index}`,
        start,
        end,
        start_time: start,
        end_time: end,
        action,
        content,
        ad_type: segment.ad_type || (action === 'skip' ? 'hard_ad' : 'mid_ad'),
        hasActionField: typeof segment.action === 'string'
      };
    }

    parseTimestampToSeconds(value) {
      if (typeof value === 'number' && Number.isFinite(value)) {
        return Math.max(0, value);
      }
      const raw = String(value ?? '').trim();
      if (!raw) return null;

      const bracketMatch = raw.match(/\[(\d{1,2}:\d{1,2}(?::\d{1,2})?)\]/);
      const source = bracketMatch ? bracketMatch[1] : raw;

      const hmsMatch = source.match(/(\d{1,2}):(\d{1,2}):(\d{1,2})/);
      if (hmsMatch) {
        const hours = Number(hmsMatch[1]);
        const minutes = Number(hmsMatch[2]);
        const seconds = Number(hmsMatch[3]);
        if ([hours, minutes, seconds].every(Number.isFinite)) {
          return Math.max(0, hours * 3600 + minutes * 60 + seconds);
        }
      }

      const msMatch = source.match(/(\d{1,3}):(\d{1,2})/);
      if (msMatch) {
        const minutes = Number(msMatch[1]);
        const seconds = Number(msMatch[2]);
        if ([minutes, seconds].every(Number.isFinite)) {
          return Math.max(0, minutes * 60 + seconds);
        }
      }

      return null;
    }

    toKnowledgeDanmuText(point) {
      if (typeof point === 'string') {
        return point.trim();
      }
      if (!point || typeof point !== 'object') {
        return '';
      }

      // 处理知识点（knowledge_points）
      const term = typeof point.term === 'string' ? point.term.trim() : '';
      const explanation = typeof point.explanation === 'string' ? point.explanation.trim() : '';
      // 处理热词（hot_words）
      const word = typeof point.word === 'string' ? point.word.trim() : '';
      const meaning = typeof point.meaning === 'string' ? point.meaning.trim() : '';

      // 知识点格式：术语: 解释
      // 热词格式：[热词] 解释
      let text = '';
      if (term && explanation) {
        text = `${term}: ${explanation}`;
      } else if (word && meaning) {
        text = `[${word}] ${meaning}`;
      } else {
        text = term || explanation || word || meaning;
      }

      if (!text) return '';
      // 不再截断文本，显示完整内容
      return text;
    }

    updateKnowledgeDanmuSource(knowledgePoints, bvid) {
      const source = Array.isArray(knowledgePoints) ? knowledgePoints : [];
      this.knowledgeDanmuQueue = source
        .map((point, index) => {
          const seconds = this.parseTimestampToSeconds(
            typeof point === 'object' ? (point.timestamp ?? point.time ?? point.start_time) : null
          );
          const text = this.toKnowledgeDanmuText(point);
          if (!Number.isFinite(seconds) || !text) return null;

          // 判断是热词还是知识点
          const isHotWord = typeof point === 'object' && point.word && point.meaning;
          const type = isHotWord ? 'hot-word' : 'knowledge-point';

          return {
            id: `${bvid || 'unknown'}-${Math.round(seconds * 10)}-${index}`,
            timeSec: seconds,
            text,
            type // 添加类型标识
          };
        })
        .filter(Boolean)
        .sort((a, b) => a.timeSec - b.timeSec);

      this.analysisBvid = bvid || this.analysisBvid;
      this.triggeredDanmuIds.clear();
      this.lastDanmuCurrentTime = 0;
      this.nextDanmuLane = 0;
      this.clearKnowledgeDanmuNodes();
    }

    clearKnowledgeDanmuNodes() {
      this.stopKnowledgeDanmuLoop();
      this.activeKnowledgeDanmus.forEach(item => {
        if (item?.node?.remove) {
          item.node.remove();
        }
      });
      this.activeKnowledgeDanmus = [];
      if (!this.knowledgeDanmuLayer) return;
      const nodes = this.knowledgeDanmuLayer.querySelectorAll('.visionmark-knowledge-danmu');
      nodes.forEach(node => node.remove());
    }

    clearKnowledgeDanmuState() {
      this.knowledgeDanmuQueue = [];
      this.triggeredDanmuIds.clear();
      this.lastDanmuCurrentTime = 0;
      this.nextDanmuLane = 0;
      this.nativeDanmuSpeedPxPerSec = 0;
      this.lastNativeSpeedSampleTs = 0;
      this.nativeDanmuTrackSample = null;
      this.clearKnowledgeDanmuNodes();
    }

    ensureKnowledgeDanmuLayer() {
      const container = document.querySelector('.bpx-player-video-wrap') ||
        document.querySelector('.bpx-player-video-area') ||
        document.querySelector('.bpx-player-container') ||
        document.querySelector('#bilibili-player');
      if (!container) return null;

      if (getComputedStyle(container).position === 'static') {
        container.style.position = 'relative';
      }

      // 加载 Noto Sans SC 字体（思源黑体 - 现代中性）
      if (!document.getElementById('visionmark-font-noto-sans')) {
        const fontLink = document.createElement('link');
        fontLink.id = 'visionmark-font-noto-sans';
        fontLink.href = 'https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400&display=swap';
        fontLink.rel = 'stylesheet';
        document.head.appendChild(fontLink);
      }

      if (!document.getElementById('visionmark-knowledge-danmu-style')) {
        const style = document.createElement('style');
        style.id = 'visionmark-knowledge-danmu-style';
        style.textContent = `
          #visionmark-knowledge-danmu-layer {
            position: absolute;
            inset: 0;
            pointer-events: none;
            overflow: hidden;
            z-index: 99998;
          }
          .visionmark-knowledge-danmu {
            position: absolute;
            max-width: min(75vw, 900px);
            min-width: 200px;
            padding: 8px 16px;

            /* 无背景 - 完全透明 */
            background: transparent !important;
            border: none !important;

            /* 增强文字效果 - 更醒目 */
            color: #ffffff !important;
            font-size: 22px;
            line-height: 1.6;
            letter-spacing: 0.5px;

            /* 使用 Noto Sans SC 字体 */
            font-family: "Noto Sans SC", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
            font-weight: 700;

            /* 多层文字阴影 - 增强可读性 */
            text-shadow:
              0 2px 4px rgba(0, 0, 0, 0.8),
              0 4px 8px rgba(0, 0, 0, 0.6),
              0 0 20px rgba(0, 0, 0, 0.5),
              0 0 40px rgba(0, 0, 0, 0.3);

            /* 允许换行，显示完整内容 */
            white-space: normal;
            word-wrap: break-word;
            word-break: break-word;

            left: 0;
            transform: translate3d(0, 0, 0);
            will-change: transform;
            transition: opacity 0.3s ease;
          }

          /* 知识点样式 - 蓝色光晕 */
          .visionmark-knowledge-danmu.knowledge-point {
            color: #66ccff !important;
            text-shadow:
              0 2px 4px rgba(0, 0, 0, 0.9),
              0 4px 8px rgba(0, 0, 0, 0.7),
              0 0 20px rgba(102, 204, 255, 0.6),
              0 0 40px rgba(102, 204, 255, 0.4);
          }

          /* 热词样式 - 粉色光晕 */
          .visionmark-knowledge-danmu.hot-word {
            color: #ff99cc !important;
            text-shadow:
              0 2px 4px rgba(0, 0, 0, 0.9),
              0 4px 8px rgba(0, 0, 0, 0.7),
              0 0 20px rgba(255, 150, 180, 0.6),
              0 0 40px rgba(255, 150, 180, 0.4);
          }
        `;
        document.head.appendChild(style);
      }

      let layer = container.querySelector('#visionmark-knowledge-danmu-layer');
      if (!layer) {
        layer = document.createElement('div');
        layer.id = 'visionmark-knowledge-danmu-layer';
        container.appendChild(layer);
      }
      this.knowledgeDanmuLayer = layer;
      return layer;
    }

    pickNativeDanmuNode() {
      const nodes = this.getVisibleNativeDanmuNodes();
      if (nodes.length) return nodes[0];
      return null;
    }

    getVisibleNativeDanmuNodes() {
      const result = [];
      const seen = new Set();
      for (const selector of DANMU_NATIVE_NODE_SELECTORS) {
        const nodes = document.querySelectorAll(selector);
        for (const node of nodes) {
          if (!(node instanceof HTMLElement)) continue;
          if (seen.has(node)) continue;
          const rect = node.getBoundingClientRect();
          if (rect.width < 8 || rect.height < 8) continue;
          if (rect.bottom <= 0 || rect.top >= window.innerHeight) continue;
          if (rect.right <= 0 || rect.left >= window.innerWidth) continue;
          seen.add(node);
          result.push(node);
        }
      }
      return result;
    }

    getKnowledgeDanmuLanes() {
      const fallback = DANMU_DEFAULT_LANES.slice(0, 3);
      if (!this.knowledgeDanmuLayer) return fallback;

      const layerRect = this.knowledgeDanmuLayer.getBoundingClientRect();
      if (!Number.isFinite(layerRect.height) || layerRect.height <= 0) return fallback;

      const nativeNodes = this.getVisibleNativeDanmuNodes();
      if (!nativeNodes.length) return fallback;

      const laneSet = new Set();
      for (const node of nativeNodes.slice(0, 24)) {
        const rect = node.getBoundingClientRect();
        const relativeY = ((rect.top + rect.height / 2) - layerRect.top) / layerRect.height * 100;
        if (!Number.isFinite(relativeY)) continue;
        const clamped = Math.min(Math.max(relativeY, DANMU_LANE_MIN_PERCENT), DANMU_LANE_MAX_PERCENT);
        const bucket = Math.round(clamped / DANMU_LANE_STEP_PERCENT) * DANMU_LANE_STEP_PERCENT;
        laneSet.add(bucket);
      }

      const nativeLanes = Array.from(laneSet)
        .filter(value => Number.isFinite(value))
        .sort((a, b) => a - b);

      const preferredLaneCount = nativeNodes.length <= 3 ? 3 : 4;
      const merged = [...nativeLanes, ...DANMU_DEFAULT_LANES]
        .filter((value, index, arr) => arr.indexOf(value) === index)
        .filter(value => value >= DANMU_LANE_MIN_PERCENT && value <= DANMU_LANE_MAX_PERCENT)
        .sort((a, b) => a - b);

      const lanes = merged.slice(0, preferredLaneCount);
      return lanes.length ? lanes : fallback;
    }

    sampleNativeDanmuSpeed(now = performance.now()) {
      if (now - this.lastNativeSpeedSampleTs < DANMU_NATIVE_SAMPLE_MS) {
        return;
      }
      this.lastNativeSpeedSampleTs = now;

      const node = this.pickNativeDanmuNode();
      if (!node) {
        this.nativeDanmuTrackSample = null;
        return;
      }

      const rect = node.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      if (!Number.isFinite(centerX)) {
        return;
      }

      const previous = this.nativeDanmuTrackSample;
      if (previous && previous.node === node) {
        const dtSec = (now - previous.ts) / 1000;
        const dx = previous.x - centerX;
        if (dtSec > 0 && dx > 0) {
          const scaleX = this.getKnowledgeDanmuLayerScaleX();
          const speed = (dx / dtSec) / scaleX;
          if (speed >= DANMU_SPEED_MIN_PX_PER_SEC && speed <= DANMU_SPEED_MAX_PX_PER_SEC) {
            this.nativeDanmuSpeedPxPerSec = this.nativeDanmuSpeedPxPerSec > 0
              ? this.nativeDanmuSpeedPxPerSec * 0.7 + speed * 0.3
              : speed;
          }
        }
      }

      this.nativeDanmuTrackSample = { node, x: centerX, ts: now };
    }

    getKnowledgeDanmuLayerScaleX() {
      if (!this.knowledgeDanmuLayer) return 1;
      const localWidth = this.knowledgeDanmuLayer.clientWidth;
      const rectWidth = this.knowledgeDanmuLayer.getBoundingClientRect().width;
      if (!Number.isFinite(localWidth) || localWidth <= 0) return 1;
      if (!Number.isFinite(rectWidth) || rectWidth <= 0) return 1;
      const scaleX = rectWidth / localWidth;
      if (!Number.isFinite(scaleX) || scaleX <= 0) return 1;
      return Math.min(Math.max(scaleX, 0.5), 3);
    }

    getFallbackDanmuSpeedPxPerSec(layerWidth) {
      const safeLocalWidth = Number.isFinite(layerWidth) && layerWidth > 0
        ? layerWidth
        : (this.knowledgeDanmuLayer?.clientWidth || window.innerWidth || 1280);
      const scaleX = this.getKnowledgeDanmuLayerScaleX();
      const viewportWidth = safeLocalWidth * scaleX;
      return ((viewportWidth + DANMU_FALLBACK_TRACK_PX) / DANMU_BASE_DURATION_SEC) / scaleX;
    }

    getKnowledgeDanmuSpeedPxPerSec(layerWidth) {
      let speed = this.getFallbackDanmuSpeedPxPerSec(layerWidth);
      if (
        Number.isFinite(this.nativeDanmuSpeedPxPerSec) &&
        this.nativeDanmuSpeedPxPerSec >= DANMU_SPEED_MIN_PX_PER_SEC &&
        this.nativeDanmuSpeedPxPerSec <= DANMU_SPEED_MAX_PX_PER_SEC
      ) {
        speed = this.nativeDanmuSpeedPxPerSec;
      }
      return speed * DANMU_SPEED_TUNE_FACTOR;
    }

    startKnowledgeDanmuLoop() {
      if (this.knowledgeDanmuRafId !== null) return;
      this.lastDanmuFrameTs = 0;
      this.knowledgeDanmuRafId = requestAnimationFrame((ts) => this.tickKnowledgeDanmu(ts));
    }

    stopKnowledgeDanmuLoop() {
      if (this.knowledgeDanmuRafId !== null) {
        cancelAnimationFrame(this.knowledgeDanmuRafId);
        this.knowledgeDanmuRafId = null;
      }
      this.lastDanmuFrameTs = 0;
    }

    syncKnowledgeDanmuAnimationState() {
      if (!this.activeKnowledgeDanmus.length) {
        this.stopKnowledgeDanmuLoop();
        return;
      }
      const state = this.player.getState();
      if (state.paused) {
        this.stopKnowledgeDanmuLoop();
        return;
      }
      this.startKnowledgeDanmuLoop();
    }

    tickKnowledgeDanmu(frameTs) {
      this.knowledgeDanmuRafId = null;
      if (!this.knowledgeDanmuLayer) {
        this.activeKnowledgeDanmus = [];
        return;
      }

      this.activeKnowledgeDanmus = this.activeKnowledgeDanmus.filter(item => item?.node?.isConnected);
      if (!this.activeKnowledgeDanmus.length) {
        this.stopKnowledgeDanmuLoop();
        return;
      }

      const state = this.player.getState();
      if (state.paused) {
        this.stopKnowledgeDanmuLoop();
        return;
      }

      const currentTs = Number.isFinite(frameTs) ? frameTs : performance.now();
      if (!this.lastDanmuFrameTs) {
        this.lastDanmuFrameTs = currentTs;
      }
      const deltaSec = Math.min(Math.max((currentTs - this.lastDanmuFrameTs) / 1000, 0), 0.1);
      this.lastDanmuFrameTs = currentTs;

      this.sampleNativeDanmuSpeed(currentTs);
      const layerWidth = this.knowledgeDanmuLayer.clientWidth || window.innerWidth || 1280;
      const speed = this.getKnowledgeDanmuSpeedPxPerSec(layerWidth);
      const remaining = [];

      for (const item of this.activeKnowledgeDanmus) {
        item.x -= speed * deltaSec;
        if (item.node && item.node.style) {
          item.node.style.transform = `translate3d(${item.x}px, 0, 0)`;
        }
        if (item.x + item.width < -DANMU_REMOVE_BUFFER_PX) {
          item.node.remove();
          continue;
        }
        remaining.push(item);
      }

      this.activeKnowledgeDanmus = remaining;
      if (!this.activeKnowledgeDanmus.length) {
        this.stopKnowledgeDanmuLoop();
        return;
      }

      this.knowledgeDanmuRafId = requestAnimationFrame((ts) => this.tickKnowledgeDanmu(ts));
    }

    renderKnowledgeDanmu(item) {
      const layer = this.ensureKnowledgeDanmuLayer();
      if (!layer) return;
      if (this.activeKnowledgeDanmus.length >= DANMU_MAX_CONCURRENT) return;

      const lanes = this.getKnowledgeDanmuLanes();
      const lane = lanes[this.nextDanmuLane % lanes.length];
      this.nextDanmuLane += 1;

      const node = document.createElement('div');
      node.className = `visionmark-knowledge-danmu ${item.type || 'knowledge-point'}`;
      node.style.top = `${lane}%`;
      node.textContent = item.text;
      layer.appendChild(node);

      const layerWidth = layer.clientWidth || window.innerWidth || 1280;
      const width = node.getBoundingClientRect().width || 240;
      const startX = layerWidth + 24;
      node.style.transform = `translate3d(${startX}px, 0, 0)`;

      this.activeKnowledgeDanmus.push({
        id: item.id,
        node,
        x: startX,
        width
      });

      this.sampleNativeDanmuSpeed();
      this.syncKnowledgeDanmuAnimationState();
    }

    handleKnowledgeDanmu(currentTime) {
      if (!Number.isFinite(currentTime) || !this.knowledgeDanmuQueue.length) {
        this.lastDanmuCurrentTime = currentTime;
        this.syncKnowledgeDanmuAnimationState();
        return;
      }
      this.sampleNativeDanmuSpeed();
      if (currentTime + DANMU_REWIND_RESET_SEC < this.lastDanmuCurrentTime) {
        this.triggeredDanmuIds.clear();
      }

      let rendered = 0;
      for (const item of this.knowledgeDanmuQueue) {
        if (rendered >= 2) break;
        if (this.triggeredDanmuIds.has(item.id)) continue;
        const delta = currentTime - item.timeSec;
        if (delta >= -DANMU_TRIGGER_WINDOW_SEC && delta <= DANMU_TRIGGER_WINDOW_SEC) {
          this.triggeredDanmuIds.add(item.id);
          this.renderKnowledgeDanmu(item);
          rendered += 1;
        }
      }

      this.lastDanmuCurrentTime = currentTime;
      this.syncKnowledgeDanmuAnimationState();
    }


    async requestAnalysis(bvid, token) {
      const url = VIDEO_ANALYSIS_BASE + "/video-analysis/analyze";
      console.log('[AdSkipper] 请求URL:', url);
      console.log('[AdSkipper] 请求体:', JSON.stringify({ bvid }));
      console.log('[AdSkipper] 注意：视频分析无超时限制，可能需要几分钟时间');

      // 直接使用原生 fetch，不设置超时
      // 视频分析需要很长时间（下载、提取、AI分析），不能有超时限制
      const res = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer ' + token
        },
        body: JSON.stringify({ bvid })
      });

      console.log('[AdSkipper] 响应状态:', res.status, res.statusText);

      let payload = null;
      try {
        payload = await res.json();
        console.log('[AdSkipper] 响应数据:', payload);
      } catch (error) {
        console.error('[AdSkipper] 解析JSON失败:', error);
        payload = null;
      }

      if (!res.ok) {
        const message = payload?.message || payload?.error || `分析失败（${res.status}）`;
        console.error('[AdSkipper] API错误:', message);
        throw new Error(message);
      }
      return payload;
    }

    applyAnalysisData(bvid, data) {
      const analysisData = data || {};
      const rawSegments = Array.isArray(analysisData.ad_segments) ? analysisData.ad_segments : [];
      const aiSegments = rawSegments
        .map((segment, index) => this.normalizeSegment({
          id: `ai-${bvid}-${index}`,
          start_time: Number(segment.start_time ?? segment.start ?? 0),
          end_time: Number(segment.end_time ?? segment.end ?? 0),
          action: segment.highlight ? 'popup' : 'skip',
          content: typeof segment.description === 'string' ? segment.description : '',
          ad_type: segment.ad_type || (segment.highlight ? 'hard_ad' : 'soft_ad'),
          is_ai_segment: true
        }, index))
        .filter(segment => Number.isFinite(segment.start_time) && Number.isFinite(segment.end_time) && segment.end_time > segment.start_time);

      const knowledgePoints = Array.isArray(analysisData.knowledge_points) ? analysisData.knowledge_points : [];
      const hotWords = Array.isArray(analysisData.hot_words) ? analysisData.hot_words : [];

      // 去重：知识点和热词重复时，以知识点为主（优先显示知识点）
      const knowledgePointTerms = new Set(knowledgePoints.map(kp => kp.term));
      const filteredHotWords = hotWords.filter(hw => !knowledgePointTerms.has(hw.word));

      // 合并知识点和过滤后的热词（知识点在前，热词在后）
      const allDanmuItems = [...knowledgePoints, ...filteredHotWords];

      console.log('[AdSkipper] 弹幕数据统计:');
      console.log('  - 知识点:', knowledgePoints.length);
      console.log('  - 原始热词:', hotWords.length);
      console.log('  - 去重后热词:', filteredHotWords.length);
      console.log('  - 总弹幕数:', allDanmuItems.length);

      this.analysisBvid = bvid;
      this.aiSummary = typeof analysisData.summary === 'string' ? analysisData.summary.trim() : '';
      this.segments = aiSegments;
      this.allSegments = aiSegments;
      this.updateKnowledgeDanmuSource(allDanmuItems, bvid);
      this.addSegmentMarkers();

      if (sidebarState) {
        sidebarState.aiSummary = this.aiSummary || '暂无总结';
        sidebarState.aiTitle = analysisData.title || '';
        sidebarState.knowledgePoints = knowledgePoints;
        sidebarState.hotWords = hotWords;
        sidebarState.bvid = bvid;
        sidebarState.cid = this.player.currentCid || null;
        sidebarState.segments = aiSegments;
        sidebarState.activeSegmentKey = null;
      }
    }

    async analyzeVideo(bvid) {
      try {
        console.log('[AdSkipper] ========== 开始视频分析 ==========');
        console.log('[AdSkipper] BV号:', bvid);
        console.log('[AdSkipper] API地址:', VIDEO_ANALYSIS_BASE);

        let token = '';
        token = await this.getToken();
        console.log('[AdSkipper] Token:', token ? '已获取（前10位: ' + token.substring(0, 10) + '...）' : '未获取');

        if (!token) {
          console.log('[AdSkipper] 未登录，跳过视频分析');
          if (sidebarState) {
            sidebarState.loadError = '请先登录后再使用AI分析功能';
          }
          return;
        }

        if (sidebarState) {
          sidebarState.isLoading = true;
          sidebarState.loadError = null;
        }

        console.log('[AdSkipper] 开始请求分析API...');
        const result = await this.requestAnalysis(bvid, token);
        console.log('[AdSkipper] API返回:', result);

        if (!result?.success || !result?.data) {
          throw new Error('分析结果无效');
        }

        this.applyAnalysisData(bvid, result.data);
        if (sidebarState) {
          sidebarState.isLoading = false;
        }
        console.log('[AdSkipper] ========== 视频分析完成 ==========');
      } catch (error) {
        console.error('[AdSkipper] 视频分析异常:', error);
        console.error('[AdSkipper] 错误详情:', {
          message: error.message,
          code: error.code,
          stack: error.stack
        });
        if (sidebarState) {
          sidebarState.isLoading = false;
          sidebarState.loadError = '分析失败: ' + error.message;
        }
      }
    }

    getSegmentKey(segment, indexFallback = 0) {
      return String(segment.id ?? `${segment.start_time}-${segment.end_time}-${indexFallback}`);
    }

    seekToSegmentStart(segment) {
      if (!segment) return;
      const targetTime = Number(segment.start_time);
      if (!Number.isFinite(targetTime)) return;
      this.player.skipTo(Math.max(0, targetTime));
    }

    seekToSegmentEnd(segment) {
      if (!segment) return;
      const targetTime = Number(segment.end_time);
      if (!Number.isFinite(targetTime)) return;
      this.player.skipTo(Math.max(0, targetTime));
    }

    getActiveSegment(currentTime) {
      return this.segments.find(segment => currentTime >= segment.start_time && currentTime < segment.end_time - 0.2);
    }

    handleSkipSegment(segment) {
      if (!segment || segment.action !== 'skip') return;
      this.seekToSegmentEnd(segment);
      this.lastSkipTime = Date.now();
      this.showSkipNotification(segment);
    }

    checkSkip(currentTime) {
      if (this.analysisBvid && this.player.currentBvid && this.analysisBvid !== this.player.currentBvid) {
        this.clearKnowledgeDanmuState();
        this.analysisBvid = this.player.currentBvid;
      }

      if (sidebarState) {
        sidebarState.currentTime = currentTime;
        if (this.player.currentBvid) {
          sidebarState.bvid = this.player.currentBvid;
        }
        if (this.player.currentCid) {
          sidebarState.cid = this.player.currentCid;
        }
      }

      this.handleKnowledgeDanmu(currentTime);

      if (!this.segments.length) {
        if (sidebarState) {
          sidebarState.activeSegmentKey = null;
        }
        return;
      }

      const activeSegment = this.getActiveSegment(currentTime);
      if (!activeSegment) {
        if (sidebarState) {
          sidebarState.activeSegmentKey = null;
        }
        return;
      }

      if (sidebarState) {
        sidebarState.activeSegmentKey = this.getSegmentKey(activeSegment);
      }

      if (activeSegment.action === 'popup') {
        return;
      }

      if (Date.now() - this.lastSkipTime < 500) {
        return;
      }

      this.handleSkipSegment(activeSegment);
    }

    showToast(msg, type) {
      const old = document.getElementById('adskipper-toast');
      if (old) old.remove();

      const t = document.createElement("div");
      t.id = 'adskipper-toast';
      t.textContent = msg;

      const color = type === 'success' ? '#67c23a' : (type === 'error' ? '#ff6b6b' : '#FB7299');
      t.style.cssText = `
        position: fixed;
        top: 15%;
        left: 50%;
        transform: translateX(-50%) translateY(-20px);
        background: ${color};
        color: #fff;
        padding: 0.8em 1.5em;
        border-radius: 0.5em;
        z-index: 999999;
        font-size: clamp(14px, 2vw, 18px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        opacity: 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      `;

      document.body.appendChild(t);

      // Animate in
      requestAnimationFrame(() => {
        t.style.opacity = '1';
        t.style.transform = 'translateX(-50%) translateY(0)';
      });

      // Animate out and remove
      setTimeout(() => {
        t.style.opacity = '0';
        t.style.transform = 'translateX(-50%) translateY(-20px)';
        setTimeout(() => t.remove(), 300);
      }, 3000);
    }

    showSkipNotification(ad) {
      const duration = (ad.end_time - ad.start_time).toFixed(1);

      // Create notification container
      const container = document.createElement('div');
      container.style.cssText = `
        position: fixed;
        top: 10%;
        left: 50%;
        transform: translateX(-50%);
        z-index: 999999;
        text-align: center;
      `;

      // Toast message
      const toast = document.createElement('div');
      toast.style.cssText = `
        background: #67c23a;
        color: #fff;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        animation: slideDown 0.4s ease-out;
      `;
      toast.textContent = `已自动跳过 ${duration}s`;

      // Progress bar
      const progress = document.createElement('div');
      progress.style.cssText = `
        width: 100%;
        height: 3px;
        background: rgba(255,255,255,0.3);
        margin-top: 8px;
        border-radius: 2px;
        overflow: hidden;
      `;

      const progressBar = document.createElement('div');
      progressBar.style.cssText = `
        height: 100%;
        background: #fff;
        width: 100%;
        animation: progress 2s linear forwards;
      `;

      progress.appendChild(progressBar);
      toast.appendChild(progress);
      container.appendChild(toast);
      document.body.appendChild(container);

      // Add animation keyframes
      if (!document.getElementById('toast-animations')) {
        const style = document.createElement('style');
        style.id = 'toast-animations';
        style.textContent = `
          @keyframes slideDown {
            0% { transform: translateY(-20px); opacity: 0; }
            100% { transform: translateY(0); opacity: 1; }
          }
          @keyframes progress {
            0% { width: 100%; }
            100% { width: 0%; }
          }
          @keyframes slideUp {
            0% { transform: translateY(0); opacity: 1; }
            100% { transform: translateY(-20px); opacity: 0; }
          }
        `;
        document.head.appendChild(style);
      }

      // Auto remove
      setTimeout(() => {
        toast.style.animation = 'slideUp 0.3s ease-out forwards';
        setTimeout(() => container.remove(), 300);
      }, 2000);
    }

    addSegmentMarkers() {
      const oldMarkers = document.querySelectorAll('.adskipper-progress-marker');
      oldMarkers.forEach(marker => marker.remove());

      if (!this.segments.length) {
        return;
      }

      const progressContainer = document.querySelector('.bpx-player-progress') ||
        document.querySelector('.bilibili-player-progress') ||
        document.querySelector('.bpx-player-progress-wrap');
      if (!progressContainer) {
        return;
      }

      const duration = this.player.getState().duration;
      if (!duration || duration <= 0) {
        return;
      }

      const progressSlide = progressContainer.querySelector('.bpx-player-progress-slide') ||
        progressContainer.querySelector('.bili-progress-slip') ||
        progressContainer.querySelector('.bpx-player-progress-buffer');
      if (!progressSlide) {
        return;
      }

      progressSlide.style.position = progressSlide.style.position || 'relative';

      this.segments.forEach((segment, index) => {
        const startPercent = (segment.start_time / duration) * 100;
        const endPercent = (segment.end_time / duration) * 100;
        const width = Math.max(endPercent - startPercent, 0.8);

        const marker = document.createElement('div');
        marker.className = 'adskipper-progress-marker';
        marker.setAttribute('data-segment-id', this.getSegmentKey(segment, index));

        const markerColor = segment.action === 'popup'
          ? 'rgba(71, 167, 255, 0.88)'
          : 'rgba(251, 114, 153, 0.82)';

        marker.style.cssText = `
          position: absolute;
          left: ${startPercent}%;
          top: 0;
          bottom: 0;
          width: ${width}%;
          background: ${markerColor} !important;
          pointer-events: none;
          z-index: 999 !important;
          height: 100% !important;
        `;

        const titleContent = segment.action === 'popup' && segment.content
          ? ` | ${segment.content.slice(0, 36)}`
          : '';
        const actionText = segment.action === 'popup' ? '重点' : '跳过';
        marker.title = `${segment.start_time.toFixed(1)}s - ${segment.end_time.toFixed(1)}s | ${actionText}${titleContent}`;

        progressSlide.appendChild(marker);
      });
    }

    handleShowMarkers() {
      this.showSidebar({ refresh: true });
    }
  }

  new AdSkipperCore().init();
})();


