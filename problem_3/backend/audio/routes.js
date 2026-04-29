/**
 * 音频分析API路由
 * 提供音频分析相关的API接口
 */

const express = require('express');
const router = express.Router();
const AudioAnalyzer = require('./analyzer');
const { authenticateToken } = require('../auth');

const analyzer = new AudioAnalyzer();

/**
 * POST /video-analysis/analyze
 * 分析单个视频（音频提取 + AI 分析）
 */
router.post('/analyze', authenticateToken, async (req, res) => {
  try {
    const { bvid } = req.body;

    if (!bvid) {
      return res.status(400).json({ error: '缺少bvid参数' });
    }

    console.log(`[API] 开始分析: ${bvid}`);

    const videoUrl = `https://www.bilibili.com/video/${bvid}`;
    const result = await analyzer.analyze(videoUrl);

    const adaptedData = {
      bvid: result.bvid,
      title: result.analysis.title,
      tags: result.analysis.tags,
      summary: result.analysis.summary,
      transcript: result.analysis.transcript,
      ad_segments: result.analysis.segments ? result.analysis.segments.map(seg => ({
        start_time: parseTimeToSeconds(seg.start_time),
        end_time: parseTimeToSeconds(seg.end_time),
        description: seg.description,
        highlight: seg.highlight,
        ad_type: seg.highlight ? 'hard_ad' : 'soft_ad'
      })) : [],
      knowledge_points: result.analysis.knowledge_points || [],
      hot_words: result.analysis.hot_words || [],
      analyzed_at: result.analyzed_at
    };

    res.json({ success: true, data: adaptedData });
  } catch (error) {
    console.error('[API] 分析失败:', error);
    res.status(500).json({
      error: '分析失败',
      message: error.message
    });
  }
});

/**
 * 将时间格式 MM:SS 或 HH:MM:SS 转换为秒数
 */
function parseTimeToSeconds(timeStr) {
  if (!timeStr) return 0;
  if (typeof timeStr === 'number') return timeStr;

  const normalizedTime = timeStr.replace(/：/g, ':');
  const parts = normalizedTime.split(':').map(p => parseFloat(p));

  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  if (parts.length === 1) return parts[0];
  return 0;
}

/**
 * POST /video-analysis/batch
 * 批量分析视频
 */
router.post('/batch', authenticateToken, async (req, res) => {
  try {
    const { videos } = req.body;

    if (!Array.isArray(videos) || videos.length === 0) {
      return res.status(400).json({ error: 'videos参数必须是非空数组' });
    }

    console.log(`[API] 开始批量分析 ${videos.length} 个视频`);

    const results = [];
    for (const video of videos) {
      try {
        const videoUrl = `https://www.bilibili.com/video/${video.bvid}`;
        const result = await analyzer.analyze(videoUrl);

        const adaptedData = {
          bvid: result.bvid,
          title: result.analysis.title,
          tags: result.analysis.tags,
          summary: result.analysis.summary,
          ad_segments: result.analysis.segments ? result.analysis.segments.map(seg => ({
            start_time: parseTimeToSeconds(seg.start_time),
            end_time: parseTimeToSeconds(seg.end_time),
            description: seg.description,
            highlight: seg.highlight,
            ad_type: seg.highlight ? 'hard_ad' : 'soft_ad'
          })) : [],
          knowledge_points: result.analysis.knowledge_points || [],
          hot_words: result.analysis.hot_words || []
        };

        results.push({ success: true, data: adaptedData });
      } catch (error) {
        results.push({ bvid: video.bvid, success: false, error: error.message });
      }
    }

    res.json({ success: true, data: results });
  } catch (error) {
    console.error('[API] 批量分析失败:', error);
    res.status(500).json({
      error: '批量分析失败',
      message: error.message
    });
  }
});

/**
 * GET /video-analysis/status/:bvid
 * 获取分析状态（待实现）
 */
router.get('/status/:bvid', authenticateToken, (req, res) => {
  res.json({
    status: 'not_implemented',
    message: '任务状态查询功能待实现'
  });
});

module.exports = router;
