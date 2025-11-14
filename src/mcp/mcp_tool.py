import os, sys
from pathlib import Path
import json
import asyncio
from agent_framework import HostedMCPTool
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
sys.path.append(str(Path(__file__).resolve().parent.parent))
from embedding.embedder import TextEmbedder
from configs.llm_config import OpenAIChatConfig

from agent_framework.openai import OpenAIChatClient, OpenAIResponsesClient, OpenAIAssistantsClient
from agent_framework import (
    ChatMessage,
    ToolProtocol,
    ChatResponse
)

class MCPTool:
    def __init__(self, llm_config: OpenAIChatConfig = None):
        if llm_config is None:
            # 使用默认的智谱AI配置
            try:
                self.llm_config = OpenAIChatConfig(model_name="zhipu")
            except ValueError as e:
                raise ValueError(f"Failed to initialize LLM config: {e}")
        else:
            self.llm_config = llm_config

        self.mcp_tools = {}

        # 添加一些默认工具
        self.add_mcp_tool(
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp",
            approval_mode="never_require",
        )

        try:
            self.chat_client = OpenAIChatClient(
                model_id=self.llm_config.model_id,
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.endpoint,
            )

            self.chat_agent = self.chat_client.create_agent(
                instruction="你是一个智能助手，你可以调用多个 MCP 工具来回答用户的问题。",
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize chat client: {e}")

    def add_mcp_tool(self, name, url, approval_mode, headers=None):
        name = name.replace(" ", "_").replace("-", "_").lower().strip()
        # 确保 name 必须只能包含英文数字下划线，否则报错
        if not name.replace("_", "").isalnum():
            raise ValueError(f"name {name} must only contain alphanumeric characters and underscores")

        self.mcp_tools[name] = HostedMCPTool(
            name=name,
            url=url,
            approval_mode=approval_mode,
            headers=headers if headers else None
        )

    async def chat_with_multiple_mcp_tools(self, query: str):
        """
        与多个 MCP 工具进行聊天
        
        Args:
            query: 用户查询字符串

        Returns:
            包含所有 MCP 工具响应的字符串
        """

        if not query or not query.strip():
            return "查询内容不能为空"

        if not self.mcp_tools:
            return "No MCP tools added. Please add MCP tools first."

        # 调用 chat_agent 进行聊天
        try:
            response = await self.chat_agent.run(
                messages=[ChatMessage(role="user", text=query)],
                tools=list(self.mcp_tools.values()),
            )
            
            # 检查response是否为空
            if not response:
                return "抱歉，AI服务没有返回有效响应"
                
            return response
            
        except Exception as e:
            return f"调用AI服务时出错: {str(e)}"

    async def test_connection(self):
        """测试与AI服务的连接"""
        try:
            test_response = await self.chat_with_multiple_mcp_tools("你好，请回复测试")
            print(f"连接测试结果: {test_response}")
            return True
        except Exception as e:
            print(f"连接测试失败: {e}")
            return False

if __name__ == "__main__":
    try:
        mcp_tool = MCPTool()
        
        # 先测试连接
        print("正在测试AI服务连接...")
        asyncio.run(mcp_tool.test_connection())
        
        # 然后进行实际对话
        print("\n开始对话测试:")
        result = asyncio.run(mcp_tool.chat_with_multiple_mcp_tools("你好, 我想知道微软的最新的 MCP 技术动态"))
        print(f"结果: {result}")
        
    except Exception as e:
        print(f"程序启动失败: {e}")
