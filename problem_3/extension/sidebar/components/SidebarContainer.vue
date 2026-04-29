<template>
  <aside
    class="vm-sidebar"
    :class="{ 'vm-sidebar--hidden': !visible }"
  >
    <SidebarHeader
      :title="title"
      :bvid="bvid"
      :segment-count="segmentCount"
      @close="handleClose"
      @refresh="handleRefresh"
    />

    <div class="vm-sidebar__content">
      <SummaryCard
        :summary="summary"
        :loading="loading"
        :error="error"
      />

      <AIAnalysisDetails
        v-if="!loading && !error"
        :title="aiTitle"
        :knowledge-points="knowledgePoints"
        :hot-words="hotWords"
      />

      <TimelineList
        :segments="segments"
        :active-key="activeKey"
        :loading="loading"
        @seek="handleSeek"
      />
    </div>
  </aside>
</template>

<script setup>
import SidebarHeader from './SidebarHeader.vue';
import SummaryCard from './SummaryCard.vue';
import AIAnalysisDetails from './AIAnalysisDetails.vue';
import TimelineList from './TimelineList.vue';

const props = defineProps({
  visible: {
    type: Boolean,
    default: false
  },
  topOffset: {
    type: Number,
    default: 64
  },
  width: {
    type: Number,
    default: 380
  },
  title: {
    type: String,
    default: '视频总结'
  },
  bvid: {
    type: String,
    default: null
  },
  segmentCount: {
    type: Number,
    default: 0
  },
  summary: {
    type: String,
    default: ''
  },
  aiTitle: {
    type: String,
    default: ''
  },
  knowledgePoints: {
    type: Array,
    default: () => []
  },
  hotWords: {
    type: Array,
    default: () => []
  },
  segments: {
    type: Array,
    default: () => []
  },
  activeKey: {
    type: String,
    default: null
  },
  loading: {
    type: Boolean,
    default: false
  },
  error: {
    type: String,
    default: null
  }
});

const emit = defineEmits(['update:visible', 'seek', 'refresh']);

function handleClose() {
  emit('update:visible', false);
}

function handleRefresh() {
  emit('refresh');
}

function handleSeek(time) {
  emit('seek', time);
}
</script>

<style scoped>
.vm-sidebar {
  position: fixed !important;
  right: 0 !important;
  top: 64px !important;
  height: calc(100vh - 64px) !important;
  width: 380px !important;
  z-index: 2147483647 !important;

  /* Crystal Glass 多层毛玻璃效果 */
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 245, 248, 0.82) 30%, rgba(255, 255, 255, 0.75) 100%),
    var(--vm-gradient-soft);
  backdrop-filter: saturate(180%) blur(28px);
  -webkit-backdrop-filter: saturate(180%) blur(28px);

  /* 精致的光影边框 */
  border-radius: 16px 0 0 16px;
  border-left: 1px solid rgba(255, 255, 255, 0.7);
  border-top: 1px solid rgba(255, 255, 255, 0.5);
  box-shadow:
    -12px 0 40px rgba(251, 114, 153, 0.06),
    -4px 0 20px rgba(0, 0, 0, 0.04),
    inset 1px 0 0 rgba(255, 255, 255, 0.9),
    inset 0 1px 0 rgba(255, 255, 255, 0.95);

  transform: translateX(0);
  transition: transform 0.3s cubic-bezier(0.2, 0.8, 0.2, 1);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  font-family: var(--vm-font-family);
}

.vm-sidebar::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 120px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.6) 0%, transparent 100%);
  pointer-events: none;
  border-radius: 16px 0 0 0;
}

.vm-sidebar--hidden {
  transform: translateX(100%);
}

.vm-sidebar__content {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
}
</style>
