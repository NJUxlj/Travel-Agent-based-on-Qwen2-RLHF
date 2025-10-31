<template>
  <div class="chat-container">
    <div class="chat-layout">
      <!-- 左侧聊天区域 -->
      <div class="chat-main">
        <!-- 聊天消息区域 -->
        <div class="chat-messages" ref="chatMessagesContainer">
          <div v-if="chatHistory.length === 0" class="empty-state">
            <div class="empty-state-icon">💬</div>
            <p>开始您的旅行规划对话吧！</p>
            <p style="font-size: 0.875rem; margin-top: 0.5rem;">您可以询问关于目的地推荐、行程规划、预算安排等问题</p>
          </div>
          <div
            v-for="(message, index) in chatHistory"
            :key="index"
            :class="['chat-message', message.role]"
          >
            <img
              :src="message.role === 'user' ? 'https://i.pravatar.cc/150?img=68' : 'https://i.pravatar.cc/150?img=33'"
              alt="Avatar"
              class="avatar"
            />
            <div class="message-content">
              {{ message.content }}
            </div>
          </div>
          <div v-if="isLoading" class="chat-message assistant">
            <img
              src="https://i.pravatar.cc/150?img=33"
              alt="Avatar"
              class="avatar"
            />
            <div class="message-content">
              <div class="loading"></div>
            </div>
          </div>
        </div>

        <!-- 输入区域 -->
        <div class="input-area">
          <div class="input-container">
            <textarea
              v-model="messageInput"
              class="textarea"
              placeholder="输入您的旅行相关问题..."
              @keydown.enter.ctrl="sendMessage"
              @keydown.enter.meta="sendMessage"
            ></textarea>
            <button
              class="btn btn-primary"
              @click="sendMessage"
              :disabled="!messageInput.trim() || isLoading"
            >
              发送
            </button>
          </div>
          <div style="margin-top: 0.5rem; display: flex; gap: 0.5rem;">
            <button class="btn btn-secondary btn-small" @click="generateMindmap">
              📊 生成思维导图
            </button>
            <button class="btn btn-secondary btn-small" @click="clearChat">
              🗑️ 清空对话
            </button>
            <span style="font-size: 0.875rem; color: var(--text-secondary); margin-left: auto;">
              Ctrl + Enter 发送
            </span>
          </div>
        </div>
      </div>

      <!-- 右侧设置和示例区域 -->
      <div class="sidebar">
        <div class="sidebar-section">
          <h3 class="sidebar-title">⚙️ 设置</h3>
          <div class="slider-container">
            <div class="slider-label">
              <span>创意度 (Temperature)</span>
              <span>{{ temperature.toFixed(1) }}</span>
            </div>
            <input
              type="range"
              class="slider"
              v-model.number="temperature"
              min="0.1"
              max="2.0"
              step="0.1"
            />
          </div>
          <div class="slider-container">
            <div class="slider-label">
              <span>精确度 (Top P)</span>
              <span>{{ topP.toFixed(1) }}</span>
            </div>
            <input
              type="range"
              class="slider"
              v-model.number="topP"
              min="0.1"
              max="1.0"
              step="0.1"
            />
          </div>
        </div>

        <div class="sidebar-section">
          <h3 class="sidebar-title">💡 示例问题</h3>
          <div class="example-buttons">
            <button
              v-for="(example, index) in examplePrompts"
              :key="index"
              class="example-btn"
              @click="useExample(example)"
            >
              {{ example }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ChatView',
  data() {
    return {
      chatHistory: [],
      messageInput: '',
      isLoading: false,
      temperature: 0.7,
      topP: 0.9,
      examplePrompts: [],
      sessionId: null
    }
  },
  mounted() {
    this.sessionId = window.travelAgentSessionId
    this.loadExamplePrompts()
    this.loadChatHistory()
  },
  methods: {
    async loadExamplePrompts() {
      try {
        const response = await fetch('/api/examples')
        if (response.ok) {
          this.examplePrompts = await response.json()
        }
      } catch (error) {
        console.error('加载示例问题失败:', error)
        // 使用默认示例
        this.examplePrompts = [
          '推荐三个适合12月份旅游的城市',
          '帮我规划一个为期3天的北京旅游行程',
          '我想去海边度假，预算8000元，有什么建议？',
          '推荐几个适合带父母旅游的目的地',
          '帮我列出去日本旅游需要准备的物品清单'
        ]
      }
    },
    
    async loadChatHistory() {
      if (!this.sessionId) return
      
      try {
        const response = await fetch(`/api/sessions/${this.sessionId}/history`)
        if (response.ok) {
          this.chatHistory = await response.json()
          this.scrollToBottom()
        }
      } catch (error) {
        console.error('加载聊天历史失败:', error)
      }
    },
    
    async sendMessage() {
      const message = this.messageInput.trim()
      if (!message || this.isLoading) return
      
      this.isLoading = true
      const tempMessage = { role: 'user', content: message }
      this.chatHistory.push(tempMessage)
      this.messageInput = ''
      this.scrollToBottom()
      
      try {
        // 使用流式响应
        const response = await fetch(`/api/sessions/${this.sessionId}/stream-messages`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            content: message,
            temperature: this.temperature,
            topP: this.topP
          })
        })
        
        if (response.ok) {
          const reader = response.body.getReader()
          const decoder = new TextDecoder()
          let fullResponse = ''
          let responseIndex = this.chatHistory.length
          
          // 添加一个空的响应消息占位
          this.chatHistory.push({ role: 'assistant', content: '' })
          
          // 流式读取响应
          while (true) {
            const { done, value } = await reader.read()
            if (done) break
            
            const chunk = decoder.decode(value)
            // 解析JSON
            const parsedChunk = JSON.parse(chunk)
            fullResponse += parsedChunk.chunk
            
            // 更新响应内容
            this.chatHistory[responseIndex].content = fullResponse
            this.scrollToBottom()
          }
        } else {
          throw new Error('请求失败')
        }
      } catch (error) {
        console.error('发送消息失败:', error)
        // 移除占位的响应消息
        this.chatHistory.pop()
        // 显示错误消息
        this.chatHistory.push({
          role: 'assistant',
          content: `抱歉，处理您的请求时发生错误：${error.message}`
        })
      } finally {
        this.isLoading = false
        this.scrollToBottom()
      }
    },
    
    async generateMindmap() {
      this.$router.push('/mindmap')
    },
    
    async clearChat() {
      if (!confirm('确定要清空所有对话记录吗？')) return
      
      try {
        const response = await fetch(`/api/sessions/${this.sessionId}/history`, {
          method: 'DELETE'
        })
        
        if (response.ok) {
          this.chatHistory = []
        }
      } catch (error) {
        console.error('清空对话失败:', error)
        alert('清空对话失败，请稍后重试')
      }
    },
    
    useExample(example) {
      this.messageInput = example
    },
    
    scrollToBottom() {
      this.$nextTick(() => {
        const container = this.$refs.chatMessagesContainer
        if (container) {
          container.scrollTop = container.scrollHeight
        }
      })
    }
  }
}
</script>

<style scoped>
.chat-container {
  height: calc(100vh - 120px);
}

.chat-layout {
  display: flex;
  height: 100%;
  gap: 1rem;
}

.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  background-color: var(--surface-color);
  border-radius: var(--border-radius);
  box-shadow: var(--shadow-light);
  overflow: hidden;
}

.chat-messages {
  flex: 1;
  padding: 1.5rem;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}

@media (max-width: 768px) {
  .chat-layout {
    flex-direction: column;
  }
  
  .sidebar {
    order: -1;
  }
  
  .chat-container {
    height: auto;
    min-height: calc(100vh - 120px);
  }
}
</style>