#!/usr/bin/env python3
"""
示例：如何在你的项目中使用 uvx 安装的 MCP 服务器
"""

import asyncio
import subprocess
import sys
from pathlib import Path

class MCPClient:
    """简单的 MCP 客户端示例"""
    
    def __init__(self):
        self.processes = {}
    
    async def start_calculator_server(self):
        """启动计算器 MCP 服务器"""
        try:
            # 使用 uvx 启动计算器服务器
            process = await asyncio.create_subprocess_exec(
                "uvx", "mcp-server-calculator",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE
            )
            self.processes['calculator'] = process
            print("✅ Calculator MCP 服务器启动成功")
        except Exception as e:
            print(f"❌ 启动计算器服务器失败: {e}")
    
    async def start_filesystem_server(self, allowed_dir="/tmp"):
        """启动文件系统 MCP 服务器"""
        try:
            # 使用 uvx 启动文件系统服务器，并限制访问目录
            process = await asyncio.create_subprocess_exec(
                "uvx", "@modelcontextprotocol/server-filesystem", allowed_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE
            )
            self.processes['filesystem'] = process
            print("✅ Filesystem MCP 服务器启动成功")
        except Exception as e:
            print(f"❌ 启动文件系统服务器失败: {e}")
    
    async def stop_all_servers(self):
        """停止所有服务器"""
        for name, process in self.processes.items():
            try:
                process.terminate()
                await process.wait()
                print(f"✅ {name} 服务器已停止")
            except Exception as e:
                print(f"❌ 停止 {name} 服务器失败: {e}")

async def main():
    """主函数演示"""
    print("🚀 启动 MCP 服务器示例")
    
    client = MCPClient()
    
    try:
        # 启动服务器
        await client.start_calculator_server()
        await client.start_filesystem_server()
        
        print("\n📝 服务器已启动，按 Ctrl+C 停止...")
        
        # 保持运行
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 收到停止信号...")
    finally:
        await client.stop_all_servers()
        print("✅ 所有服务器已停止")

if __name__ == "__main__":
    # 检查 uvx 是否可用
    try:
        result = subprocess.run(["uvx", "--help"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ uvx 可用，开始运行示例...")
            asyncio.run(main())
        else:
            print("❌ uvx 不可用，请先安装 uv")
            sys.exit(1)
    except FileNotFoundError:
        print("❌ uvx 未找到，请先安装 uv: curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)