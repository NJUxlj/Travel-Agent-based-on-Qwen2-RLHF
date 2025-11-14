import asyncio
import os, sys
from typing import Optional
from contextlib import AsyncExitStack
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic, AsyncClient
from dotenv import load_dotenv

from openai import AsyncOpenAI
from agent_framework.openai import OpenAIChatClient, OpenAIResponsesClient, OpenAIAssistantsClient
from agent_framework import (
    ChatMessage,
    ToolProtocol,
    ChatResponse
)

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.llm_config import OpenAIChatConfig

load_dotenv()  # load environment variables from .env


class MCPToolImplementation:
    """A concrete implementation of ToolProtocol for MCP tools."""
    
    def __init__(self, name: str, description: str, input_schema: dict):
        self.name = name
        self.description = description
        self.additional_properties = input_schema

class MCPClient:
    def __init__(self, llm_config: OpenAIChatConfig = OpenAIChatConfig()):
        # Initialize session and client objects
        self.llm_config = llm_config
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

        self.chat_client = AsyncOpenAI(
            api_key = llm_config.api_key,
            base_url = llm_config.endpoint,
        )
    


    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        # 构造 MCP 服务器启动参数：指定用 python 或 node 启动，传入脚本路径，不额外设置环境变量
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        # 第1行：建立与MCP服务器的stdio传输连接
        # 使用stdio_client创建与服务器的标准输入/输出通信通道
        # server_params包含了启动服务器的详细信息（命令类型、脚本路径、环境变量等）
        # exit_stack确保连接在会话结束时能够正确清理资源
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        
        # 第2行：从stdio传输中解包获取读写通道
        # stdio：标准输入/输出流对象，用于从MCP服务器读取响应数据
        # write：写入流对象，用于向MCP服务器发送请求数据
        self.stdio, self.write = stdio_transport
        
        # 第3行：创建MCP客户端会话对象
        # ClientSession是MCP协议的核心会话类，处理与服务器的所有通信
        # 使用stdio读取通道和write写入通道建立双向通信
        # 将session加入exit_stack管理，确保会话结束时资源正确释放
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])



    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()

        available_tools = []


        for tool in response.tools:
            tool_schema = getattr(
                tool,
                "inputSchema",
                {"type": "object", "properties": {}, "required": []},
            )

            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool_schema,
                },
            }
            available_tools.append(openai_tool)


        # Initial Claude API call
        model_response = await self.chat_client.chat.completions.create(
            model=self.llm_config.model_id,
            max_tokens=1000,
            messages=messages,
            tools=available_tools,
        )

        # Process response and handle tool calls
        tool_results = []
        final_text = []

        messages.append(model_response.choices[0].message.model_dump())
        print(messages[-1])
        if model_response.choices[0].message.tool_calls:
            tool_call = model_response.choices[0].message.tool_calls[0]
            tool_args = json.loads(tool_call.function.arguments)

            tool_name = tool_call.function.name
            result = await self.session.call_tool(tool_name, tool_args)
            tool_results.append({"call": tool_name, "result": result})
            final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

            messages.append(
                {
                    "role": "tool",
                    "content": f"{result}",
                    "tool_call_id": tool_call.id,
                }
            )

            # Get next response from Claude
            response = await self.chat_client.chat.completions.create(
                model=self.llm_config.model_id,
                max_tokens=1000,
                messages=messages,
            )

            messages.append(response.choices[0].message.model_dump())
            print(messages[-1])

        return messages[-1]["content"]




    
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()




async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())