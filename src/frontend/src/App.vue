<template>
  <header class="header">
    <div class="container header-content">
      <div class="logo">
        🌍 AI Travel Agent
      </div>
      <div>
        <router-link to="/" class="btn btn-secondary btn-small">
          💬 对话
        </router-link>
        <router-link to="/mindmap" class="btn btn-secondary btn-small" style="margin-left: 0.5rem;">
          📊 思维导图
        </router-link>
      </div>
    </div>
  </header>
  <main class="container" style="flex: 1; padding: 1.5rem 0;">
    <router-view v-slot="{ Component }">
      <transition name="fade" mode="out-in">
        <component :is="Component" />
      </transition>
    </router-view>
  </main>
  <footer style="text-align: center; padding: 1rem 0; color: var(--text-secondary); border-top: 1px solid var(--border-color);">
    <div class="container">
      <p>© 2024 AI Travel Agent | 基于Qwen2模型的智能旅行助手</p>
    </div>
  </footer>
</template>

<script>
export default {
  name: 'App',
  data() {
    return {
      sessionId: null
    }
  },
  mounted() {
    // 检查本地存储中是否已有会话ID，没有则创建新会话
    this.initSession()
  },
  methods: {
    async initSession() {
      let sessionId = localStorage.getItem('travelAgentSessionId')
      
      if (!sessionId) {
        try {
          const response = await fetch('/api/sessions', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            }
          })
          
          if (response.ok) {
            const data = await response.json()
            sessionId = data.session_id
            localStorage.setItem('travelAgentSessionId', sessionId)
          }
        } catch (error) {
          console.error('创建会话失败:', error)
        }
      }
      
      this.sessionId = sessionId
      // 设置为全局属性，方便其他组件访问
      window.travelAgentSessionId = sessionId
    }
  }
}
</script>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>