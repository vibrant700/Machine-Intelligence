// 使用配置文件中的 API 地址（如果配置文件存在）
const API_BASE = window.LOCAL_CONFIG
  ? window.LOCAL_CONFIG.API_BASE + '/' + window.LOCAL_CONFIG.API_VERSION
  : 'http://localhost:8080/api/v1';

const NETWORK_ERROR_CODE = 'NETWORK_UNAVAILABLE';
const NETWORK_COOLDOWN_MS = 30000;
const NETWORK_TIMEOUT_MS = 6000;
const USER_CACHE_TTL_MS = 5000;

const networkState = {
  offlineUntil: 0,
  hasLoggedOffline: false,
  wasOffline: false
};

const userInfoCache = {
  data: null,
  updatedAt: 0,
  pending: null
};

function createNetworkUnavailableError() {
  const error = new Error('网络不可用，请稍后重试');
  error.code = NETWORK_ERROR_CODE;
  return error;
}

function isNetworkFailure(error) {
  if (!error) return false;
  if (error.name === 'AbortError') return true;
  const message = String(error.message || '').toLowerCase();
  if (message.includes('failed to fetch') || message.includes('networkerror') || message.includes('load failed')) {
    return true;
  }
  return error instanceof TypeError;
}

function markNetworkOffline(error) {
  networkState.offlineUntil = Date.now() + NETWORK_COOLDOWN_MS;
  networkState.wasOffline = true;
  if (!networkState.hasLoggedOffline) {
    console.warn('[Popup] 后端不可达，暂停请求 30 秒。', error);
    networkState.hasLoggedOffline = true;
  }
}

function markNetworkOnline() {
  if (networkState.wasOffline) {
    console.info('[Popup] 后端连接已恢复');
  }
  networkState.offlineUntil = 0;
  networkState.hasLoggedOffline = false;
  networkState.wasOffline = false;
}

async function safeFetch(url, options = {}) {
  if (Date.now() < networkState.offlineUntil) {
    throw createNetworkUnavailableError();
  }

  const controller = options.signal ? null : new AbortController();
  const timeoutId = setTimeout(() => {
    if (controller) controller.abort();
  }, NETWORK_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      ...options,
      signal: options.signal || (controller ? controller.signal : undefined)
    });
    clearTimeout(timeoutId);
    markNetworkOnline();
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (isNetworkFailure(error)) {
      markNetworkOffline(error);
      throw createNetworkUnavailableError();
    }
    throw error;
  }
}

function resetUserInfoCache() {
  userInfoCache.data = null;
  userInfoCache.updatedAt = 0;
  userInfoCache.pending = null;
}

function formatTierLabel(tier) {
  const rawTier = String(tier || '').trim();
  if (!rawTier) return '青铜会员';

  const normalized = rawTier.toLowerCase();
  const tierMap = {
    bronze: '青铜会员',
    silver: '白银会员',
    gold: '黄金会员',
    platinum: '铂金会员',
    diamond: '钻石会员',
    admin: '管理员'
  };

  if (tierMap[normalized]) return tierMap[normalized];
  if (/[一-龥]/.test(rawTier)) return rawTier;
  return '普通会员';
}

async function apiRequest(endpoint, options = {}) {
  try {
    const token = localStorage.getItem('adskipper_token');
    const headers = {
      'Content-Type': 'application/json',
      ...options.headers
    };
    if (token) headers['Authorization'] = 'Bearer ' + token;

    const res = await safeFetch(API_BASE + endpoint, {
      ...options,
      headers
    });

    const contentType = res.headers.get('content-type');
    if (!contentType || !contentType.includes('application/json')) {
      throw new Error('服务器返回格式错误（可能后端未启动）');
    }

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || '请求失败');
    return data;
  } catch(err) {
    if (err.code !== NETWORK_ERROR_CODE) {
      console.error('API Error:', err);
    }
    throw err;
  }
}

async function checkAuth() {
  const token = localStorage.getItem('adskipper_token');
  if (!token) {
    showLoginForm();
    return false;
  }

  try {
    const user = JSON.parse(localStorage.getItem('adskipper_user') || '{}');
    if (user.username) {
      showUserPanel(user);
      return true;
    }
  } catch(e) {
    localStorage.removeItem('adskipper_token');
    localStorage.removeItem('adskipper_user');
  }
  showLoginForm();
  return false;
}

async function getCurrentUserInfo(forceRefresh = false) {
  const now = Date.now();
  if (!forceRefresh && userInfoCache.data && (now - userInfoCache.updatedAt) < USER_CACHE_TTL_MS) {
    return userInfoCache.data;
  }

  if (!forceRefresh && userInfoCache.pending) {
    return userInfoCache.pending;
  }

  userInfoCache.pending = apiRequest('/auth/me')
    .then((data) => {
      userInfoCache.data = data;
      userInfoCache.updatedAt = Date.now();
      return data;
    })
    .finally(() => {
      userInfoCache.pending = null;
    });

  return userInfoCache.pending;
}

function showLoginForm() {
  document.getElementById('auth-form').style.display = 'block';
  document.getElementById('user-panel').style.display = 'none';
}

async function showUserPanel(user) {
  document.getElementById('auth-form').style.display = 'none';
  document.getElementById('user-panel').style.display = 'block';
  document.getElementById('user-panel').classList.add('user-panel-active');

  document.getElementById('display-username').textContent = user.username;
  document.getElementById('display-points').textContent = user.points || 0;
  document.getElementById('display-tier').textContent = formatTierLabel(user.tier);
}

async function refreshUserInfo() {
  try {
    const user = await getCurrentUserInfo(true);
    localStorage.setItem('adskipper_user', JSON.stringify(user));
    document.getElementById('display-points').textContent = user.points || 0;
    document.getElementById('display-tier').textContent = formatTierLabel(user.tier);
    console.log('[Popup] 积分已刷新:', user.points);
  } catch(err) {
    if (err.code !== NETWORK_ERROR_CODE) {
      console.error('[Popup] 刷新用户信息失败:', err);
    }
  }
}

function showError(msg) {
  const err = document.getElementById('error-msg');
  err.textContent = msg;
  err.style.display = 'block';
  setTimeout(() => err.style.display = 'none', 4000);
}

async function handleAuth() {
  const username = document.getElementById('username').value.trim();
  const password = document.getElementById('password').value;
  const isLogin = document.getElementById('submit-btn').textContent === '登录';

  if (!username || !password) {
    showError('请填写用户名和密码');
    return;
  }

  const btn = document.getElementById('submit-btn');
  const originalText = btn.textContent;
  btn.disabled = true;
  btn.textContent = isLogin ? '登录中...' : '注册中...';

  try {
    const endpoint = isLogin ? '/auth/login' : '/auth/register';
    const data = await apiRequest(endpoint, {
      method: 'POST',
      body: JSON.stringify({ username, password })
    });

    if (isLogin) {
      localStorage.setItem('adskipper_token', data.token);
      localStorage.setItem('adskipper_user', JSON.stringify({
        username: data.username,
        points: data.points || 0,
        tier: data.tier || 'bronze',
        userId: data.userId || null
      }));
      chrome.storage.local.set({ adskipper_token: data.token }, () => {
        console.log('[Popup] Token已同步到chrome.storage.local');
      });
      userInfoCache.data = data;
      userInfoCache.updatedAt = Date.now();
      userInfoCache.pending = null;
      showUserPanel(data);
    } else {
      showError('✓ 注册成功，请登录');
      toggleMode();
    }
  } catch(err) {
    showError(err.message || '网络错误，请检查后端是否启动 (localhost:8080)');
  } finally {
    btn.disabled = false;
    btn.textContent = originalText;
  }
}

function toggleMode() {
  const btn = document.getElementById('submit-btn');
  const switchText = document.getElementById('switch-text');
  const isLogin = btn.textContent === '登录';

  if (isLogin) {
    btn.textContent = '注册';
    switchText.textContent = '已有账号？登录';
  } else {
    btn.textContent = '登录';
    switchText.textContent = '还没有账号？立即注册';
  }
}

function logout() {
  localStorage.removeItem('adskipper_token');
  localStorage.removeItem('adskipper_user');
  resetUserInfoCache();
  chrome.storage.local.remove(['adskipper_token']);
  showLoginForm();
}

// 初始化
document.addEventListener('DOMContentLoaded', async () => {
  const isLoggedIn = await checkAuth();

  if (isLoggedIn) {
    refreshUserInfo();
  }

  document.getElementById('submit-btn').onclick = handleAuth;
  document.getElementById('switch-text').onclick = toggleMode;
  document.getElementById('logout-btn').onclick = logout;

  document.getElementById('password').onkeypress = (e) => {
    if (e.key === 'Enter') handleAuth();
  };

  apiRequest('/health')
    .then(() => console.log('后端连接正常'))
    .catch(() => showError('警告：无法连接后端，请确保localhost:8080运行中'));
});
