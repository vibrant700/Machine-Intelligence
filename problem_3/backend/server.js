require('dotenv').config();

const express = require('express');
const cors = require('cors');
const config = require('./config.js');
const { authRoutes } = require('./auth');
const audioAnalysisRouter = require('./audio/routes');

const app = express();

app.use(cors({ origin: config.CORS_ORIGIN, credentials: true }));
app.use(express.json());

// 注册路由
authRoutes(app).then(() => {
  app.use('/video-analysis', audioAnalysisRouter);

  app.get('/api/v1/health', (req, res) => res.json({ ok: true }));

  app.listen(config.PORT, '0.0.0.0', () => {
    console.log('[Server] http://localhost:' + config.PORT);
    console.log('[Auth] admin/admin 登录');
  });
});
