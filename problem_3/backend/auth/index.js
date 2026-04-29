const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const fs = require('fs');
const path = require('path');
const config = require('../config.js');

const initSqlJs = require('sql.js');

const DB_PATH = path.join(__dirname, 'database', 'app.db');
let db = null;

async function getDb() {
  if (db) return db;

  if (!fs.existsSync(path.dirname(DB_PATH))) {
    fs.mkdirSync(path.dirname(DB_PATH), { recursive: true });
  }

  const SQL = await initSqlJs();
  if (fs.existsSync(DB_PATH)) {
    const buffer = fs.readFileSync(DB_PATH);
    db = new SQL.Database(buffer);
  } else {
    db = new SQL.Database();
  }

  db.run(`
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT UNIQUE NOT NULL,
      password_hash TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS user_points (
      user_id INTEGER PRIMARY KEY,
      total_points INTEGER DEFAULT 0,
      tier TEXT DEFAULT 'bronze'
    );
  `);

  // 创建测试账号
  const result = db.exec("SELECT * FROM users WHERE username = 'admin'");
  if (!result.length || result[0].values.length === 0) {
    const hash = bcrypt.hashSync('admin', 10);
    db.run("INSERT INTO users (username, password_hash) VALUES (?, ?)", ['admin', hash]);
    const idResult = db.exec("SELECT id FROM users WHERE username = 'admin'");
    const userId = idResult[0].values[0][0];
    db.run("INSERT INTO user_points (user_id, total_points, tier) VALUES (?, 999, 'platinum')", [userId]);
    saveDb();
    console.log('[Auth] 测试账号: admin/admin');
  }

  return db;
}

function saveDb() {
  if (!db) return;
  const data = db.export();
  const buffer = Buffer.from(data);
  fs.writeFileSync(DB_PATH, buffer);
}

const JWT_SECRET = config.JWT_SECRET;

function authenticateToken(req, res, next) {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json({ error: '未登录，请先在插件中登录' });
  }

  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) {
      if (err.name === 'TokenExpiredError') {
        return res.status(403).json({ error: '登录已过期，请重新登录' });
      }
      return res.status(403).json({ error: 'Token无效，请重新登录' });
    }
    req.user = user;
    next();
  });
}

async function authRoutes(app) {
  // 确保数据库在启动时初始化
  await getDb();

  app.post('/api/v1/auth/login', (req, res) => {
    const { username, password } = req.body;
    const result = db.exec("SELECT * FROM users WHERE username = ?", [username]);

    if (!result.length || result[0].values.length === 0) {
      return res.status(401).json({ error: '用户名或密码错误' });
    }

    const row = result[0].values[0];
    const user = { id: row[0], username: row[1], password_hash: row[2] };

    if (!bcrypt.compareSync(password, user.password_hash)) {
      return res.status(401).json({ error: '用户名或密码错误' });
    }

    const token = jwt.sign({ userId: user.id, username }, JWT_SECRET, { expiresIn: '7d' });
    const pointsResult = db.exec("SELECT * FROM user_points WHERE user_id = ?", [user.id]);
    const points = pointsResult.length ? { total_points: pointsResult[0].values[0][1], tier: pointsResult[0].values[0][2] } : null;

    res.json({
      token,
      username,
      points: points ? points.total_points : 0,
      tier: points ? points.tier : 'bronze'
    });
  });

  app.post('/api/v1/auth/register', (req, res) => {
    const { username, password } = req.body;
    if (!username || !password) return res.status(400).json({ error: '必填' });

    const existing = db.exec("SELECT id FROM users WHERE username = ?", [username]);
    if (existing.length && existing[0].values.length > 0) return res.status(409).json({ error: '已存在' });

    const hash = bcrypt.hashSync(password, 10);
    db.run("INSERT INTO users (username, password_hash) VALUES (?, ?)", [username, hash]);
    const idResult = db.exec("SELECT id FROM users WHERE username = ?", [username]);
    const userId = idResult[0].values[0][0];
    db.run("INSERT INTO user_points (user_id) VALUES (?)", [userId]);
    saveDb();

    res.json({ message: '注册成功' });
  });

  app.get('/api/v1/auth/me', authenticateToken, (req, res) => {
    const result = db.exec("SELECT * FROM user_points WHERE user_id = ?", [req.user.userId]);
    const points = result.length ? { total_points: result[0].values[0][1], tier: result[0].values[0][2] } : null;
    res.json({
      username: req.user.username,
      userId: req.user.userId,
      points: points ? points.total_points : 0,
      tier: points ? points.tier : 'bronze'
    });
  });
}

module.exports = { authenticateToken, authRoutes };
