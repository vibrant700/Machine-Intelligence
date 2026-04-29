const axios = require('axios');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const util = require('util');
const ffmpegPath = require('@ffmpeg-installer/ffmpeg').path;
const OpenAI = require('openai');

const execPromise = util.promisify(exec);
const PYTHON_PATH = process.env.PYTHON_PATH;
if (!PYTHON_PATH) {
  throw new Error('请在 .env 文件中配置 PYTHON_PATH');
}

// 通义千问API配置 - 使用OpenAI兼容模式
const QWEN_API_KEY = process.env.QWEN_API_KEY;
if (!QWEN_API_KEY) {
  console.warn('[AudioAnalyzer] QWEN_API_KEY 未配置，AI分析功能将不可用');
}

// 创建OpenAI客户端
const openaiClient = new OpenAI({
  apiKey: QWEN_API_KEY,
  baseURL: 'https://dashscope.aliyuncs.com/compatible-mode/v1'
});

// 阿里云OSS配置 - 用于上传音频文件
const OSS = require('ali-oss');

const hasOssConfig = Boolean(
  process.env.OSS_ACCESS_KEY_ID &&
  process.env.OSS_ACCESS_KEY_SECRET &&
  process.env.OSS_BUCKET
);

let ossClient = null;
if (hasOssConfig) {
  ossClient = new OSS({
    region: process.env.OSS_REGION || 'oss-cn-beijing',
    accessKeyId: process.env.OSS_ACCESS_KEY_ID,
    accessKeySecret: process.env.OSS_ACCESS_KEY_SECRET,
    bucket: process.env.OSS_BUCKET
  });
} else {
  console.warn('[AudioAnalyzer] OSS config missing, audio transcription upload will be skipped.');
}

class AudioAnalyzer {
  constructor() {
    this.downloadDir = path.join(__dirname, 'downloads');
    this.ensureDownloadDir();
  }

  normalizeUrl(input) {
    if (!input) return '';
    return String(input).trim().replace(/^[`'"]+|[`'"]+$/g, '').replace(/\s+/g, '');
  }

  ensureDownloadDir() {
    if (!fs.existsSync(this.downloadDir)) {
      fs.mkdirSync(this.downloadDir, { recursive: true });
    }
  }

  /**
   * 从B站URL提取视频信息
   */
  extractBilibiliInfo(url) {
    const normalizedUrl = this.normalizeUrl(url);
    const bvMatch = normalizedUrl.match(/BV[\w]+/i);
    if (bvMatch) {
      return { bvid: bvMatch[0], url: normalizedUrl || url };
    }
    throw new Error('无法从URL中提取BV号');
  }

  /**
   * 使用yt-dlp仅下载音频
   */
  async downloadAudio(bvid, url) {
    const wavPath = path.join(this.downloadDir, `${bvid}.wav`);

    // 检查是否已存在
    if (fs.existsSync(wavPath)) {
      console.log(`[AudioAnalyzer] 音频已存在: ${wavPath}`);
      return wavPath;
    }

    // 检查是否有旧的mp3
    const oldMp3Path = path.join(this.downloadDir, `${bvid}.mp3`);
    if (fs.existsSync(oldMp3Path)) {
      console.log(`[AudioAnalyzer] 找到旧的MP3音频，将使用: ${oldMp3Path}`);
      return oldMp3Path;
    }

    console.log(`[AudioAnalyzer] 开始下载音频 ${bvid}...`);

    try {
      const normalizedUrl = this.normalizeUrl(url);

      const outputTemplate = path.join(this.downloadDir, `${bvid}.%(ext)s`);
      const command = `"${PYTHON_PATH}" -m yt_dlp --ffmpeg-location "${ffmpegPath}" -f "bestaudio[ext=m4a]/bestaudio" -x --audio-format wav --audio-quality 0 --postprocessor-args "ffmpeg:-ar 16000 -ac 1" -o "${outputTemplate}" "${normalizedUrl}"`;

      await execPromise(command, { shell: true });

      // 查找下载的音频文件
      const audioFiles = fs.readdirSync(this.downloadDir).filter(
        f => f.startsWith(bvid) && (f.endsWith('.wav') || f.endsWith('.m4a') || f.endsWith('.mp3') || f.endsWith('.opus'))
      );

      if (audioFiles.length === 0) {
        throw new Error('音频下载完成但找不到文件');
      }

      const audioFilePath = path.join(this.downloadDir, audioFiles[0]);
      console.log(`[AudioAnalyzer] 音频下载完成: ${audioFilePath}`);

      // 如果yt-dlp没有正确转为WAV，手动转换
      if (!audioFiles[0].endsWith('.wav')) {
        console.log(`[AudioAnalyzer] 转换音频为WAV格式...`);
        const convertCommand = `"${ffmpegPath}" -i "${audioFilePath}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "${wavPath}" -y`;
        await execPromise(convertCommand, { shell: true });
        // 删除原始格式文件
        fs.unlinkSync(audioFilePath);
        console.log(`[AudioAnalyzer] 音频转换完成: ${wavPath}`);
        return wavPath;
      }

      return audioFilePath;
    } catch (error) {
      console.error('[AudioAnalyzer] 音频下载失败:', error);
      throw new Error(`音频下载失败: ${error.message}`);
    }
  }

  /**
   * 使用通义千问语音识别进行音频转录（paraformer-v2异步API）
   * @param {string} audioPath - 音频文件路径
   * @param {string} bvid - 视频BV号
   */
  async transcribeAudio(audioPath, bvid) {
    console.log('[AudioAnalyzer] 开始语音识别...');

    try {
      if (!ossClient) {
        console.warn('[AudioAnalyzer] OSS client unavailable, skip transcription.');
        return null;
      }

      // 检查文件大小
      const stats = fs.statSync(audioPath);
      const fileSizeMB = stats.size / (1024 * 1024);

      if (fileSizeMB > 100) {
        console.warn(`[AudioAnalyzer] 音频文件过大(${fileSizeMB.toFixed(2)}MB)，跳过语音识别`);
        return null;
      }

      console.log(`[AudioAnalyzer] 音频文件大小: ${fileSizeMB.toFixed(2)}MB`);

      // 1. 上传音频到阿里云OSS
      console.log('[AudioAnalyzer] 上传音频到OSS...');
      const ossObjectName = `audio/${bvid}/${path.basename(audioPath)}`;

      try {
        const result = await ossClient.put(ossObjectName, audioPath);
        console.log('[AudioAnalyzer] 音频已上传到OSS:', result.url);
      } catch (error) {
        console.error('[AudioAnalyzer] OSS上传失败:', error.message);
        throw new Error(`音频上传OSS失败: ${error.message}`);
      }

      // 2. 使用paraformer-v2异步API进行语音识别
      console.log('[AudioAnalyzer] 使用 paraformer-v2 异步API 进行语音识别...');

      const audioUrl = `https://${process.env.OSS_BUCKET}.${process.env.OSS_REGION}.aliyuncs.com/${ossObjectName}`;

      // Step 1: 提交异步任务
      console.log('[AudioAnalyzer] 提交语音识别任务...');
      const submitResponse = await axios.post(
        'https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription',
        {
          model: 'paraformer-v2',
          input: {
            file_urls: [audioUrl]
          },
          parameters: {
            text_mode: 'sentence',
            language_hints: ['zh', 'en'],
            disfluency_removal: false,
            timestamp_alignment: true
          }
        },
        {
          headers: {
            'Authorization': `Bearer ${QWEN_API_KEY}`,
            'Content-Type': 'application/json',
            'X-DashScope-Async': 'enable'
          }
        }
      );

      if (!submitResponse.data.output || !submitResponse.data.output.task_id) {
        throw new Error('提交任务失败，未获取到task_id');
      }

      const taskId = submitResponse.data.output.task_id;
      console.log('[AudioAnalyzer] 任务已提交，task_id:', taskId);

      // Step 2: 轮询任务结果
      console.log('[AudioAnalyzer] 等待任务完成...');
      const maxAttempts = 60;
      let attempts = 0;

      while (attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        attempts++;

        try {
          const resultResponse = await axios.get(
            `https://dashscope.aliyuncs.com/api/v1/tasks/${taskId}`,
            {
              headers: {
                'Authorization': `Bearer ${QWEN_API_KEY}`,
                'Content-Type': 'application/json'
              }
            }
          );

          const taskStatus = resultResponse.data.output?.task_status;

          console.log(`[AudioAnalyzer] 任务状态: ${taskStatus} (${attempts}/${maxAttempts})`);

          if (taskStatus === 'SUCCEEDED') {

            if (resultResponse.data.output?.results && resultResponse.data.output.results.length > 0) {
              const firstResult = resultResponse.data.output.results[0];

              if (firstResult.transcription_url) {
                console.log('[AudioAnalyzer] 检测到 transcription_url，正在下载转录结果...');
                try {
                  const transcriptionResponse = await axios.get(firstResult.transcription_url);
                  const transcriptionData = transcriptionResponse.data;

                  let transcriptText = '';

                  function splitByCommas(sentences) {
                    const result = [];
                    sentences.forEach(sentence => {
                      const beginTime = (sentence.begin_time || 0) / 1000;
                      const text = sentence.text || '';

                      const parts = text.split('，');
                      const endTime = (sentence.end_time || sentence.begin_time || 0) / 1000;
                      const duration = endTime - beginTime;

                      parts.forEach((part, index) => {
                        if (part.trim()) {
                          const partTime = beginTime + (duration * index / parts.length);
                          const minutes = Math.floor(partTime / 60);
                          const seconds = Math.floor(partTime % 60);
                          const timeStr = `${minutes}:${seconds.toString().padStart(2, '0')}`;
                          result.push(`[${timeStr}] ${part.trim()}`);
                        }
                      });
                    });
                    return result.join('\n');
                  }

                  if (transcriptionData.transcripts && transcriptionData.transcripts.length > 0) {
                    console.log('[AudioAnalyzer] 使用 transcripts 数据结构（逗号分割模式）');
                    const allSentences = transcriptionData.transcripts.flatMap(t => t.sentences || []);
                    transcriptText = splitByCommas(allSentences);
                  }
                  else if (transcriptionData.transcription_lines) {
                    console.log('[AudioAnalyzer] 使用 transcription_lines 数据结构（逗号分割模式）');
                    transcriptText = splitByCommas(transcriptionData.transcription_lines);
                  } else if (Array.isArray(transcriptionData)) {
                    console.log('[AudioAnalyzer] 使用数组数据结构（逗号分割模式）');
                    transcriptText = splitByCommas(transcriptionData);
                  } else if (typeof transcriptionData === 'string') {
                    console.log('[AudioAnalyzer] 使用字符串数据结构');
                    transcriptText = transcriptionData;
                  }

                  if (!transcriptText) {
                    console.warn('[AudioAnalyzer] 无法解析转录数据结构');
                  }

                  console.log('[AudioAnalyzer] 语音识别完成（从transcription_url下载）');
                  console.log('[AudioAnalyzer] 转录内容预览:', transcriptText.substring(0, 500).replace(/\n/g, ' '));
                  return transcriptText;
                } catch (error) {
                  console.error('[AudioAnalyzer] 下载转录结果失败:', error.message);
                  return null;
                }
              } else if (firstResult.transcription_text) {
                const transcript = resultResponse.data.output.results
                  .map(result => {
                    const time = (result.begin_time || result.timestamp || result.start_time || result.time || 0) / 1000;
                    const minutes = Math.floor(time / 60);
                    const seconds = Math.floor(time % 60);
                    const timeStr = `${minutes}:${seconds.toString().padStart(2, '0')}`;
                    return `[${timeStr}] ${result.transcription_text}`;
                  })
                  .join('\n');

                console.log('[AudioAnalyzer] 语音识别完成');
                console.log('[AudioAnalyzer] 转录内容预览:', transcript.substring(0, 500).replace(/\n/g, ' '));
                return transcript;
              }
            }
            console.warn('[AudioAnalyzer] 任务成功但没有返回转录结果');
            return null;
          } else if (taskStatus === 'FAILED') {
            throw new Error('语音识别任务失败: ' + JSON.stringify(resultResponse.data.output?.message));
          } else if (taskStatus === 'RUNNING' || taskStatus === 'PENDING') {
            continue;
          } else {
            throw new Error('未知任务状态: ' + taskStatus);
          }
        } catch (error) {
          if (error.response) {
            throw new Error(`查询任务状态失败: ${JSON.stringify(error.response.data)}`);
          }
          throw error;
        }
      }

      throw new Error('语音识别任务超时');
    } catch (error) {
      console.error('[AudioAnalyzer] 语音识别失败:', error.response?.data || error.message);
      return null;
    }
  }

  /**
   * 从音频中提取知识点
   * @param {string} transcript - 音频转录文本
   */
  async extractKnowledgePoints(transcript) {
    if (!transcript) return null;

    console.log('[AudioAnalyzer] 提取知识点...');

    try {
      const response = await openaiClient.chat.completions.create({
        model: 'qwen-turbo',
        messages: [
          {
            role: 'user',
            content: `请从以下文本中提取重要的知识点、概念和术语，进行学霸提示和百科解读：

文本内容（可能包含[MM:SS]时间标记）：
${transcript}

请以JSON格式返回：
{
  "knowledge_points": [
    {
      "term": "术语/概念名称",
      "explanation": "详细解释说明",
      "type": "知识点类型（如：技术概念/历史知识/科学原理等）",
      "timestamp": "出现时间点(格式必须为MM:SS)。如果文本中有[MM:SS]标记，请直接使用该标记；否则请根据上下文推算。"
    }
  ]
}

提取3-8个最重要的知识点。请务必标注每个知识点在文本中大致出现的时间点（根据文本顺序或[MM:SS]标记推测）。`
          }
        ],
        max_tokens: 2000
      });

      if (response && response.choices && response.choices[0]) {
        const result = response.choices[0].message.content;
        const jsonMatch = result.match(/```json\n([\s\S]*?)\n```/) || result.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          const jsonStr = jsonMatch[1] || jsonMatch[0];
          return JSON.parse(jsonStr);
        }
      }
      return null;
    } catch (error) {
      console.error('[AudioAnalyzer] 知识点提取失败:', error.message);
      return null;
    }
  }

  /**
   * 识别热词和网络梗（仅提取转录文本中原有的词）
   * @param {string} transcript - 音频转录文本
   */
  async extractHotWords(transcript) {
    if (!transcript) return null;

    console.log('[AudioAnalyzer] 识别热词和梗...');

    try {
      const response = await openaiClient.chat.completions.create({
        model: 'qwen-turbo',
        messages: [
          {
            role: 'user',
            content: `请从以下转录文本中提取网络热词、流行梗和饭圈用语。

**重要要求：只能提取转录文本中原有的词汇，不能自己编造或解释新的词！**

转录文本内容（格式：[MM:SS] 文本内容）：
${transcript}

请以JSON格式返回：
{
  "hot_words": [
    {
      "word": "从转录文本中直接提取的热词（必须是原文中出现的词）",
      "meaning": "简要解释这个词的含义",
      "category": "分类（如：网络梗/流行语/饭圈用语等）",
      "timestamp": "出现时间点(格式必须为MM:SS)。必须直接使用转录文本中的[MM:SS]标记。"
    }
  ]
}

**注意**：
1. 只能提取转录文本中实际出现的词
2. 不要创造或添加文本中没有的词
3. 必须使用转录文本中的[MM:SS]时间戳
4. 如果某个词在转录文本中没有明确的时间戳，就不要提取它
5. 提取3-5个最热门的词`
          }
        ],
        max_tokens: 2000
      });

      if (response && response.choices && response.choices[0]) {
        const result = response.choices[0].message.content;
        const jsonMatch = result.match(/```json\n([\s\S]*?)\n```/) || result.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          const jsonStr = jsonMatch[1] || jsonMatch[0];
          return JSON.parse(jsonStr);
        }
      }
      return null;
    } catch (error) {
      console.error('[AudioAnalyzer] 热词识别失败:', error.message);
      return null;
    }
  }

  /**
   * 使用qwen-turbo基于转录文本进行综合分析（替代原Qwen-VL视觉分析）
   * @param {string} transcript - 音频转录文本
   */
  async analyzeWithText(transcript) {
    console.log('[AudioAnalyzer] 调用通义千问 qwen-turbo 进行文本分析...');

    const promptText = `请作为一个资深B站用户和百科全书，根据以下语音转录内容对视频进行深度分析。

ASR语音转录内容（格式：[MM:SS] 文本内容）：
${transcript || '无语音内容'}

**注意**：上述转录文本中的 [MM:SS] 是精确的时间戳，例如 [0:15] 表示该内容在视频第15秒出现。

请输出JSON格式报告：
{
  "title": "视频标题（若未知可根据内容生成）",
  "tags": ["标签1", "标签2"],
  "summary": "300字以内的视频精彩总结，包含核心看点",
  "segments": [
    {
      "start_time": "MM:SS",
      "end_time": "MM:SS",
      "description": "该片段的核心内容概要",
      "highlight": true/false (是否高能片段)
    }
  ]
}

**重要要求**：
1. 分段的start_time和end_time必须直接使用转录文本中的[MM:SS]时间戳
2. 不要推测没有时间戳的时间点
3. summary要生动有趣，适合B站用户口味
4. highlight标记真正有价值的高能时刻`;

    try {
      const completion = await openaiClient.chat.completions.create({
        model: 'qwen-turbo',
        messages: [
          {
            role: 'user',
            content: promptText
          }
        ],
        max_tokens: 3000,
        timeout: 120000
      });

      if (completion && completion.choices && completion.choices[0]) {
        const aiResponse = completion.choices[0].message.content;
        console.log('[AudioAnalyzer] AI分析完成');

        try {
          const jsonMatch = aiResponse.match(/```json\n([\s\S]*?)\n```/) ||
                           aiResponse.match(/\{[\s\S]*\}/);

          if (jsonMatch) {
            const jsonStr = jsonMatch[1] || jsonMatch[0];
            const parsed = JSON.parse(jsonStr);

            const result = {
              title: parsed.title || '未知标题',
              tags: parsed.tags || [],
              summary: parsed.summary || '',
              segments: parsed.segments || [],
              raw_response: aiResponse
            };

            console.log('[AudioAnalyzer] JSON解析成功');
            console.log('[AudioAnalyzer] - segments:', result.segments.length);

            return result;
          }

          console.warn('[AudioAnalyzer] 未找到JSON格式，返回原始响应');
          return {
            title: '解析失败',
            tags: [],
            summary: aiResponse.substring(0, 200),
            segments: [],
            raw_response: aiResponse,
            parse_error: '无法提取JSON格式的分析结果'
          };
        } catch (parseError) {
          console.error('[AudioAnalyzer] JSON解析失败:', parseError);
          return {
            title: '解析失败',
            tags: [],
            summary: aiResponse ? aiResponse.substring(0, 200) : '解析错误',
            segments: [],
            raw_response: aiResponse,
            parse_error: parseError.message
          };
        }
      } else {
        throw new Error('API返回数据格式错误');
      }
    } catch (error) {
      console.error('[AudioAnalyzer] API调用失败:', error.response?.data || error.message);
      throw new Error(`AI分析失败: ${error.message}`);
    }
  }

  /**
   * 完整的音频分析流程
   */
  async analyze(url) {
    try {
      // 1. 提取BV号
      const { bvid } = this.extractBilibiliInfo(url);
      console.log(`[AudioAnalyzer] 开始分析: ${bvid}`);

      // 2. 下载音频（仅音频，不下载视频）
      const audioPath = await this.downloadAudio(bvid, url);

      // 3. 语音识别
      let transcript = null;
      try {
        transcript = await this.transcribeAudio(audioPath, bvid);
      } catch (error) {
        console.warn('[AudioAnalyzer] 音频处理失败:', error.message);
      }

      // 4. 基于文本的综合分析（标题、摘要、分段）
      const analysisResult = await this.analyzeWithText(transcript);

      // 5. 并行提取知识点和热词
      console.log('[AudioAnalyzer] 并行提取知识点和热词...');
      const [kpResult, hwResult] = await Promise.all([
        this.extractKnowledgePoints(transcript),
        this.extractHotWords(transcript)
      ]);
      console.log(`[AudioAnalyzer] 知识点: ${kpResult?.knowledge_points?.length || 0}个, 热词: ${hwResult?.hot_words?.length || 0}个`);

      // 6. 整合结果
      const finalResult = {
        ...analysisResult,
        knowledge_points: kpResult?.knowledge_points || [],
        hot_words: hwResult?.hot_words || [],
        transcript: transcript
      };

      console.log(`[AudioAnalyzer] 分析完成: ${bvid}`);

      return {
        bvid,
        analysis: finalResult,
        analyzed_at: new Date().toISOString()
      };
    } catch (error) {
      console.error('[AudioAnalyzer] 分析失败:', error);
      throw error;
    }
  }

  /**
   * 清理下载的音频文件
   */
  cleanup(bvid) {
    try {
      const files = fs.readdirSync(this.downloadDir);
      const audioFiles = files.filter(f => f.startsWith(bvid) && (f.endsWith('.wav') || f.endsWith('.mp3') || f.endsWith('.m4a') || f.endsWith('.opus')));

      audioFiles.forEach(file => {
        const filePath = path.join(this.downloadDir, file);
        fs.unlinkSync(filePath);
        console.log(`[AudioAnalyzer] 已删除文件: ${filePath}`);
      });
    } catch (error) {
      console.error('[AudioAnalyzer] 清理文件失败:', error);
    }
  }
}

module.exports = AudioAnalyzer;
