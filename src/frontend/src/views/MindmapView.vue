<template>
  <div class="mindmap-container">
    <div class="mindmap-header">
      <h1>📊 旅行计划思维导图</h1>
      <div>
        <button class="btn btn-primary" @click="generateMindmap">
          🆕 生成新思维导图
        </button>
        <button class="btn btn-secondary" style="margin-left: 0.5rem;" @click="backToChat">
          ↩️ 返回对话
        </button>
      </div>
    </div>

    <div v-if="isLoading" class="loading-container">
      <div class="loading-spinner"></div>
      <p>正在生成思维导图...</p>
    </div>

    <div v-else-if="!mindmapData" class="empty-state">
      <div class="empty-state-icon">📋</div>
      <p>还没有思维导图数据</p>
      <p style="font-size: 0.875rem; margin-top: 0.5rem;">请先在对话中获取旅行建议，然后生成思维导图</p>
      <button class="btn btn-primary" style="margin-top: 1rem;" @click="generateMindmap">
        生成思维导图
      </button>
    </div>

    <div v-else class="mindmap-content">
      <div class="mindmap-controls">
        <button class="btn btn-secondary btn-small" @click="zoomIn">
          🔍 放大
        </button>
        <button class="btn btn-secondary btn-small" @click="zoomOut">
          🔍 缩小
        </button>
        <button class="btn btn-secondary btn-small" @click="resetZoom">
          🔄 重置缩放
        </button>
        <select v-model="layoutType" class="select" @change="updateLayout">
          <option value="tree">树形布局</option>
          <option value="radial">放射状布局</option>
          <option value="force">力导向布局</option>
        </select>
      </div>

      <div class="mindmap-visualization" ref="mindmapContainer" :style="{ transform: `scale(${zoomLevel})`, transformOrigin: 'center center' }">
        <div v-if="layoutType === 'tree'" class="tree-layout">
          <div class="tree-node root-node">
            <div class="node-content">{{ mindmapData.title }}</div>
            <div class="tree-children">
              <div v-for="(section, index) in mindmapData.sections" :key="index" class="tree-node">
                <div class="node-content">{{ section.title }}</div>
                <div class="tree-children">
                  <div v-for="(item, itemIndex) in section.items" :key="itemIndex" class="tree-node">
                    <div class="node-content">{{ item }}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div v-else-if="layoutType === 'radial'" class="radial-layout" ref="radialContainer">
          <!-- 放射状布局将通过JavaScript实现 -->
        </div>

        <div v-else-if="layoutType === 'force'" class="force-layout" ref="forceContainer">
          <!-- 力导向布局将通过JavaScript实现 -->
        </div>
      </div>

      <div class="mindmap-info">
        <h3>旅行信息摘要</h3>
        <div class="info-grid">
          <div class="info-item">
            <strong>目的地:</strong> {{ mindmapData.destination || '未指定' }}
          </div>
          <div class="info-item">
            <strong>天数:</strong> {{ mindmapData.days || '未指定' }}
          </div>
          <div class="info-item">
            <strong>预算:</strong> {{ mindmapData.budget || '未指定' }}
          </div>
          <div class="info-item">
            <strong>最佳时间:</strong> {{ mindmapData.bestTime || '未指定' }}
          </div>
        </div>
        
        <div class="summary-section" v-if="mindmapData.summary">
          <h4>旅行概要</h4>
          <p>{{ mindmapData.summary }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'MindmapView',
  data() {
    return {
      mindmapData: null,
      isLoading: false,
      zoomLevel: 1,
      layoutType: 'tree',
      sessionId: null
    }
  },
  mounted() {
    this.sessionId = window.travelAgentSessionId
    this.loadMindmap()
  },
  watch: {
    mindmapData() {
      this.$nextTick(() => {
        if (this.layoutType === 'radial') {
          this.renderRadialLayout()
        } else if (this.layoutType === 'force') {
          this.renderForceLayout()
        }
      })
    }
  },
  methods: {
    async loadMindmap() {
      if (!this.sessionId) return
      
      try {
        const response = await fetch(`/api/sessions/${this.sessionId}/mindmap`)
        if (response.ok) {
          this.mindmapData = await response.json()
        }
      } catch (error) {
        console.error('加载思维导图失败:', error)
      }
    },
    
    async generateMindmap() {
      if (!this.sessionId) {
        alert('请先创建会话')
        return
      }
      
      this.isLoading = true
      
      try {
        const response = await fetch(`/api/sessions/${this.sessionId}/generate-mindmap`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          }
        })
        
        if (response.ok) {
          this.mindmapData = await response.json()
        } else {
          throw new Error('生成思维导图失败')
        }
      } catch (error) {
        console.error('生成思维导图失败:', error)
        alert('生成思维导图失败，请稍后重试')
      } finally {
        this.isLoading = false
      }
    },
    
    backToChat() {
      this.$router.push('/')
    },
    
    zoomIn() {
      if (this.zoomLevel < 2) {
        this.zoomLevel += 0.1
      }
    },
    
    zoomOut() {
      if (this.zoomLevel > 0.5) {
        this.zoomLevel -= 0.1
      }
    },
    
    resetZoom() {
      this.zoomLevel = 1
    },
    
    updateLayout() {
      this.$nextTick(() => {
        if (this.layoutType === 'radial') {
          this.renderRadialLayout()
        } else if (this.layoutType === 'force') {
          this.renderForceLayout()
        }
      })
    },
    
    renderRadialLayout() {
      // 简单的放射状布局实现
      const container = this.$refs.radialContainer
      if (!container || !this.mindmapData) return
      
      container.innerHTML = ''
      
      // 创建根节点
      const rootNode = document.createElement('div')
      rootNode.className = 'radial-node root-node'
      rootNode.style.position = 'absolute'
      rootNode.style.top = '50%'
      rootNode.style.left = '50%'
      rootNode.style.transform = 'translate(-50%, -50%)'
      rootNode.innerHTML = `<div class="node-content">${this.mindmapData.title}</div>`
      container.appendChild(rootNode)
      
      // 创建子节点
      const sectionCount = this.mindmapData.sections.length
      const radius = 200
      
      this.mindmapData.sections.forEach((section, index) => {
        const angle = (2 * Math.PI * index) / sectionCount
        const x = 50 + radius * Math.cos(angle)
        const y = 50 + radius * Math.sin(angle)
        
        const sectionNode = document.createElement('div')
        sectionNode.className = 'radial-node'
        sectionNode.style.position = 'absolute'
        sectionNode.style.top = `${y}%`
        sectionNode.style.left = `${x}%`
        sectionNode.style.transform = 'translate(-50%, -50%)'
        sectionNode.innerHTML = `<div class="node-content">${section.title}</div>`
        container.appendChild(sectionNode)
      })
    },
    
    renderForceLayout() {
      // 简单的力导向布局模拟
      // 实际应用中可以考虑使用d3.js等库
      const container = this.$refs.forceContainer
      if (!container || !this.mindmapData) return
      
      container.innerHTML = ''
      
      // 创建所有节点并随机分布
      const nodes = []
      
      // 根节点
      const rootNode = document.createElement('div')
      rootNode.className = 'force-node root-node'
      rootNode.style.left = '50%'
      rootNode.style.top = '50%'
      rootNode.innerHTML = `<div class="node-content">${this.mindmapData.title}</div>`
      container.appendChild(rootNode)
      nodes.push(rootNode)
      
      // 子节点
      this.mindmapData.sections.forEach((section, index) => {
        const sectionNode = document.createElement('div')
        sectionNode.className = 'force-node'
        sectionNode.style.left = `${Math.random() * 80 + 10}%`
        sectionNode.style.top = `${Math.random() * 80 + 10}%`
        sectionNode.innerHTML = `<div class="node-content">${section.title}</div>`
        container.appendChild(sectionNode)
        nodes.push(sectionNode)
      })
    }
  }
}
</script>

<style scoped>
.mindmap-container {
  height: calc(100vh - 120px);
  display: flex;
  flex-direction: column;
}

.mindmap-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.mindmap-content {
  flex: 1;
  display: flex;
  gap: 1rem;
  overflow: hidden;
}

.mindmap-visualization {
  flex: 1;
  background-color: var(--surface-color);
  border-radius: var(--border-radius);
  box-shadow: var(--shadow-light);
  padding: 2rem;
  overflow: auto;
  position: relative;
  min-height: 400px;
}

.mindmap-info {
  width: 300px;
  background-color: var(--surface-color);
  border-radius: var(--border-radius);
  box-shadow: var(--shadow-light);
  padding: 1.5rem;
  overflow-y: auto;
}

.tree-layout {
  display: flex;
  justify-content: center;
}

.tree-node {
  position: relative;
  padding: 1rem;
  text-align: center;
}

.tree-children {
  display: flex;
  justify-content: center;
  margin-top: 1.5rem;
  flex-wrap: wrap;
}

.tree-children .tree-node {
  margin: 0 1rem;
  position: relative;
}

.tree-children .tree-node::before {
  content: '';
  position: absolute;
  top: -1rem;
  left: 50%;
  width: 2px;
  height: 1rem;
  background-color: var(--border-color);
}

.node-content {
  padding: 0.75rem 1rem;
  border-radius: var(--border-radius);
  background-color: var(--primary-color);
  color: white;
  font-weight: 500;
  box-shadow: var(--shadow-light);
}

.root-node .node-content {
  background-color: var(--secondary-color);
  padding: 1rem 1.5rem;
  font-size: 1.125rem;
}

.radial-layout,
.force-layout {
  position: relative;
  width: 100%;
  height: 500px;
}

.radial-node,
.force-node {
  position: absolute;
}

.mindmap-controls {
  margin-bottom: 1rem;
  display: flex;
  gap: 0.5rem;
  align-items: center;
}

.info-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 0.75rem;
  margin: 1rem 0;
}

.summary-section {
  margin-top: 1.5rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border-color);
}

@media (max-width: 768px) {
  .mindmap-content {
    flex-direction: column;
  }
  
  .mindmap-info {
    width: 100%;
  }
}
</style>