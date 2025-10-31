import { createRouter, createWebHistory } from 'vue-router'
import ChatView from '../views/ChatView.vue'
import MindmapView from '../views/MindmapView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'chat',
      component: ChatView,
      meta: { title: 'AI旅行助手 - 对话' }
    },
    {
      path: '/mindmap',
      name: 'mindmap',
      component: MindmapView,
      meta: { title: 'AI旅行助手 - 思维导图' }
    }
  ]
})

// 全局前置守卫，设置页面标题
router.beforeEach((to, from, next) => {
  document.title = to.meta.title || 'AI旅行助手'
  next()
})

export default router