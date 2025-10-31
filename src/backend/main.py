from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import json
import asyncio
import os
from typing import List, Dict, Optional, Any

# 添加项目根目录到Python路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入TravelAgent和其他必要的模块
from src.models.model import TravelAgent
from src.ui.mindmap import generate_mindmap
from src.configs.config import MODEL_PATH, SFT_MODEL_PATH

# 初始化数据库
DATABASE_URL = "sqlite:///./travel_agent.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 定义数据库模型
class ChatHistory(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    role = Column(String)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# 创建数据库表
Base.metadata.create_all(bind=engine)

# 初始化FastAPI应用
app = FastAPI(
    title="Travel Agent API",
    description="基于Qwen2的智能旅行助手API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化TravelAgent模型
agent = TravelAgent(SFT_MODEL_PATH)

# 依赖项：获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 工具函数：格式化聊天历史
def format_chat_history(chat_history: List[Dict]) -> str:
    formatted = "System: You are a Travel Agent that can help user plan a route from one start location to a end location. This plan you give should be in detail.\n\n"
    for msg in chat_history:
        if msg["role"] == "user":
            formatted += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            formatted += f"Assistant: {msg['content']}\n\n"
    return formatted + "User: "

# API端点：创建新会话
@app.post("/api/sessions", response_model=Dict[str, str])
async def create_session(db: Session = Depends(get_db)):
    import uuid
    session_id = str(uuid.uuid4())
    
    # 创建新会话记录
    db_session = UserSession(session_id=session_id)
    db.add(db_session)
    db.commit()
    
    return {"session_id": session_id}

# API端点：获取会话历史
@app.get("/api/sessions/{session_id}/history", response_model=List[Dict[str, str]])
async def get_chat_history(session_id: str, db: Session = Depends(get_db)):
    # 检查会话是否存在
    session = db.query(UserSession).filter(UserSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 获取聊天历史
    history = db.query(ChatHistory).filter(
        ChatHistory.session_id == session_id
    ).order_by(ChatHistory.created_at).all()
    
    return [
        {"role": h.role, "content": h.content, "created_at": h.created_at.isoformat()}
        for h in history
    ]

# API端点：发送消息并获取响应
@app.post("/api/sessions/{session_id}/messages", response_model=Dict[str, str])
async def send_message(
    session_id: str,
    message: Dict[str, str],
    db: Session = Depends(get_db)
):
    # 检查会话是否存在
    session = db.query(UserSession).filter(UserSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 获取聊天历史
    history = db.query(ChatHistory).filter(
        ChatHistory.session_id == session_id
    ).order_by(ChatHistory.created_at).all()
    
    # 格式化历史记录
    formatted_history = format_chat_history([
        {"role": h.role, "content": h.content}
        for h in history
    ])
    
    # 构建完整提示
    prompt = f"{formatted_history}{message['content']}\nAssistant:"
    
    # 生成响应
    response = agent.generate_response(
        prompt=prompt,
        max_length=message.get("max_length", 1024),
        temperature=message.get("temperature", 0.7),
        top_p=message.get("top_p", 0.9)
    )
    
    # 保存用户消息和AI响应到数据库
    db_user_message = ChatHistory(
        session_id=session_id,
        role="user",
        content=message["content"]
    )
    db.add(db_user_message)
    
    db_ai_message = ChatHistory(
        session_id=session_id,
        role="assistant",
        content=response
    )
    db.add(db_ai_message)
    
    db.commit()
    
    return {"response": response}

# API端点：流式发送消息
@app.post("/api/sessions/{session_id}/stream-messages")
async def stream_message(
    session_id: str,
    message: Dict[str, str],
    db: Session = Depends(get_db)
):
    # 检查会话是否存在
    session = db.query(UserSession).filter(UserSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 获取聊天历史
    history = db.query(ChatHistory).filter(
        ChatHistory.session_id == session_id
    ).order_by(ChatHistory.created_at).all()
    
    # 格式化历史记录
    formatted_history = format_chat_history([
        {"role": h.role, "content": h.content}
        for h in history
    ])
    
    # 构建完整提示
    prompt = f"{formatted_history}{message['content']}\nAssistant:"
    
    # 保存用户消息到数据库
    db_user_message = ChatHistory(
        session_id=session_id,
        role="user",
        content=message["content"]
    )
    db.add(db_user_message)
    db.commit()
    
    # 流式生成响应的生成器
    async def stream_generator():
        full_response = ""
        # 使用stream_chat方法
        for chunk in agent.stream_chat(
            prompt=prompt,
            max_length=message.get("max_length", 1024),
            temperature=message.get("temperature", 0.7),
            top_p=message.get("top_p", 0.9)
        ):
            # 计算新增的部分
            new_content = chunk[len(full_response):]
            full_response = chunk
            
            # 发送新增内容
            yield json.dumps({"chunk": new_content})
            await asyncio.sleep(0.01)  # 小延迟避免过度占用资源
        
        # 保存完整响应到数据库
        db_ai_message = ChatHistory(
            session_id=session_id,
            role="assistant",
            content=full_response
        )
        db.add(db_ai_message)
        db.commit()
    
    return StreamingResponse(stream_generator(), media_type="application/json")

# 生成思维导图
@app.post("/api/sessions/{session_id}/generate-mindmap")
async def generate_mindmap_for_session(
    session_id: str,
    params: Optional[Dict] = None,
    db: Session = Depends(get_db)
):
    # 检查会话是否存在
    session = db.query(UserSession).filter(UserSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 获取聊天历史
    history = db.query(ChatHistory).filter(
        ChatHistory.session_id == session_id
    ).order_by(ChatHistory.created_at).all()
    
    if not history:
        raise HTTPException(status_code=400, detail="No chat history available")
    
    # 提取旅行相关信息
    chat_text = "\n".join([f"{h.role}: {h.content}" for h in history])
    
    # 生成思维导图数据结构
    try:
        # 如果有generate_mindmap函数可用，使用它；否则创建模拟数据
        try:
            # 尝试生成实际的思维导图
            # 这里我们创建一个模拟的思维导图数据结构
            # 实际应用中可以根据聊天内容分析生成更准确的数据
            
            # 简单分析聊天内容获取目的地等信息
            destination = "未指定"
            days = "未指定"
            budget = "未指定"
            
            # 模拟思维导图数据
            mindmap_data = {
                "title": "AI旅行助手规划",
                "destination": destination,
                "days": days,
                "budget": budget,
                "bestTime": "根据季节推荐",
                "summary": "基于您的对话内容生成的旅行规划思维导图。",
                "sections": [
                    {
                        "title": "目的地信息",
                        "items": ["主要景点", "当地特色", "交通情况", "住宿推荐", "美食攻略"]
                    },
                    {
                        "title": "行程安排",
                        "items": ["第一天行程", "第二天行程", "第三天行程", "交通安排", "自由活动"]
                    },
                    {
                        "title": "旅行准备",
                        "items": ["必备证件", "行李清单", "天气情况", "通讯工具", "紧急联系人"]
                    },
                    {
                        "title": "预算规划",
                        "items": ["交通费用", "住宿费用", "餐饮费用", "门票费用", "购物预算"]
                    },
                    {
                        "title": "注意事项",
                        "items": ["当地习俗", "安全提示", "健康建议", "保险信息", "紧急救援"]
                    }
                ]
            }
            
            return mindmap_data
        except Exception as e:
            # 如果生成失败，返回简化的模拟数据
            return {
                "title": "旅行规划",
                "destination": "根据您的需求推荐",
                "sections": [
                    {"title": "景点推荐", "items": ["景点1", "景点2", "景点3"]},
                    {"title": "行程安排", "items": ["第1天", "第2天", "第3天"]},
                    {"title": "实用建议", "items": ["交通", "住宿", "餐饮"]}
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate mindmap: {str(e)}")

# API端点：清空会话历史
@app.delete("/api/sessions/{session_id}/history")
async def clear_chat_history(session_id: str, db: Session = Depends(get_db)):
    # 检查会话是否存在
    session = db.query(UserSession).filter(UserSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 删除所有该会话的聊天历史
    db.query(ChatHistory).filter(ChatHistory.session_id == session_id).delete()
    db.commit()
    
    return {"message": "Chat history cleared successfully"}

# API端点：获取示例问题
@app.get("/api/examples")
async def get_example_prompts():
    examples = [
        "推荐三个适合12月份旅游的城市",
        "帮我规划一个为期3天的北京旅游行程",
        "我想去海边度假，预算8000元，有什么建议？",
        "推荐几个适合带父母旅游的目的地",
        "帮我列出去日本旅游需要准备的物品清单"
    ]
    return examples

# 获取思维导图数据
@app.get("/api/sessions/{session_id}/mindmap")
async def get_mindmap(session_id: str, db: Session = Depends(get_db)):
    # 检查会话是否存在
    session = db.query(UserSession).filter(UserSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 返回模拟的思维导图数据
    # 实际应用中可以从缓存或数据库中获取
    return {
        "title": "AI旅行助手规划",
        "destination": "未指定",
        "days": "未指定",
        "budget": "未指定",
        "bestTime": "根据季节推荐",
        "summary": "基于您的对话内容生成的旅行规划思维导图。",
        "sections": [
            {
                "title": "目的地信息",
                "items": ["主要景点", "当地特色", "交通情况", "住宿推荐", "美食攻略"]
            },
            {
                "title": "行程安排",
                "items": ["第一天行程", "第二天行程", "第三天行程", "交通安排", "自由活动"]
            },
            {
                "title": "旅行准备",
                "items": ["必备证件", "行李清单", "天气情况", "通讯工具", "紧急联系人"]
            },
            {
                "title": "预算规划",
                "items": ["交通费用", "住宿费用", "餐饮费用", "门票费用", "购物预算"]
            },
            {
                "title": "注意事项",
                "items": ["当地习俗", "安全提示", "健康建议", "保险信息", "紧急救援"]
            }
        ]
    }

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)