# AI旅行助手前端应用

## 项目概述

本项目是一个基于Vue.js构建的智能旅行助手前端应用，为用户提供与基于Qwen2模型的旅行代理系统交互的现代化界面。用户可以通过聊天界面获取旅行建议，并使用思维导图功能可视化旅行规划。

## 技术框架

### 前端技术栈
- **框架**：Vue 3 + JavaScript
- **构建工具**：Vite
- **路由**：Vue Router
- **HTTP客户端**：Axios (内置fetch API)
- **样式**：CSS3 (自定义主题系统)

### 后端技术栈
- **框架**：FastAPI (Python)
- **数据库**：SQLite
- **模型**：TravelAgent (基于Qwen2)

## 架构设计

### 前端架构

前端采用单页面应用(SPA)架构，主要包含以下模块：

1. **路由管理**：使用Vue Router处理页面导航，包含对话页面和思维导图页面
2. **状态管理**：基于会话ID的本地状态管理，使用localStorage存储会话信息
3. **API交互**：封装了与后端FastAPI服务的所有交互
4. **UI组件**：
   - 对话界面：支持流式消息显示、参数调节、示例问题
   - 思维导图：支持多种布局展示（树形、放射状、力导向）

### 核心功能流程

1. **用户会话管理**：
   - 首次访问自动创建新会话
   - 会话ID存储在localStorage中
   - 支持持久化聊天历史

2. **聊天交互流程**：
   - 用户发送消息
   - 后端接收并处理
   - 前端显示流式响应
   - 自动保存聊天记录

3. **思维导图生成**：
   - 基于聊天历史生成结构化数据
   - 提供多种可视化布局
   - 支持缩放和交互操作

## 功能特性

- **智能对话**：与TravelAgent模型进行交互式聊天，支持流式响应
- **参数调节**：可调整temperature和top_p参数控制回答风格
- **思维导图**：基于聊天内容生成旅行规划思维导图，提供三种不同布局
- **响应式设计**：适配不同设备的屏幕尺寸
- **会话管理**：自动创建和维护用户会话
- **示例问题**：提供常见旅行问题作为参考

## 目录结构

```
src/frontend/
├── public/               # 静态资源
├── src/                  # 源代码
│   ├── assets/           # 资源文件
│   │   └── main.css      # 主样式文件
│   ├── components/       # Vue组件
│   ├── router/           # 路由配置
│   │   └── index.js      # 路由定义
│   ├── views/            # 页面视图
│   │   ├── ChatView.vue  # 聊天界面
│   │   └── MindmapView.vue # 思维导图界面
│   ├── App.vue           # 根组件
│   └── main.js           # 入口文件
├── .env                  # 环境变量
├── index.html            # HTML入口
├── package.json          # 项目配置
├── vite.config.js        # Vite配置
└── README.md             # 项目说明
```

## 安装和运行步骤

### 前提条件

- Node.js >= 16.x
- npm >= 8.x 或 yarn >= 1.x
- 后端服务正在运行 (http://localhost:8000)

### 安装步骤

1. **克隆项目**（如果尚未克隆）

2. **进入前端目录**
   ```bash
   cd /Users/xiniuyiliao/Desktop/code/Travel-Agent-based-on-Qwen2-RLHF/src/frontend
   ```

3. **安装依赖**
   ```bash
   npm install
   # 或使用yarn
   # yarn install
   ```

### 运行项目

#### 开发模式

启动开发服务器，支持热重载：

```bash
npm run dev
# 或使用yarn
# yarn dev
```

开发服务器默认运行在 http://localhost:3000

#### 构建生产版本

```bash
npm run build
# 或使用yarn
# yarn build
```

构建后的文件将输出到 `dist` 目录。

#### 预览生产版本

```bash
npm run preview
# 或使用yarn
# yarn preview
```

## 环境配置

项目使用以下环境变量（配置在`.env`文件中）：

- `VITE_API_BASE_URL`: 后端API基础URL（默认为http://localhost:8000/api）
- `NODE_ENV`: 构建模式（development或production）

## 开发指南

### 添加新页面

1. 在 `src/views/` 创建新的Vue组件
2. 在 `src/router/index.js` 中注册新路由
3. 更新导航组件以链接到新页面

### 修改主题

主题变量定义在 `src/assets/main.css` 文件中，可以通过修改CSS变量来调整颜色、字体等样式。

### 与后端交互

前端通过以下主要API端点与后端交互：

- `POST /api/sessions`: 创建新会话
- `GET /api/sessions/{session_id}/history`: 获取聊天历史
- `POST /api/sessions/{session_id}/stream-messages`: 流式发送消息
- `POST /api/sessions/{session_id}/generate-mindmap`: 生成思维导图
- `GET /api/examples`: 获取示例问题

## 故障排除

### 常见问题

1. **无法连接后端**
   - 确保后端服务正在运行
   - 检查API_BASE_URL配置是否正确

2. **思维导图不显示**
   - 确保有足够的聊天历史
   - 检查浏览器控制台是否有错误信息

3. **流式响应不工作**
   - 检查网络连接
   - 确认后端是否支持流式返回

## 许可证

MIT License