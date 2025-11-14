# MCP 服务器安装和使用指南

## 概述

MCP (Model Context Protocol) 是一个标准协议，允许大型语言模型安全地与外部工具和数据源交互。使用 `uvx` 可以轻松安装和使用各种 MCP 服务器。

## 安装 uv 和 uvx

首先确保你已安装 `uv` 包管理器：

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 Homebrew (macOS)
brew install uv

# 添加到 PATH (如果需要)
export PATH="$HOME/.local/bin:$PATH"
```

验证安装：
```bash
uv --version
uvx --help
```

## 热门 MCP 服务器列表

### 1. 计算器服务器 (Calculator)

**安装命令：**
```bash
uvx mcp-server-calculator
```

**功能：**
- 提供数学计算能力
- 支持复杂数学表达式评估
- 帮助 AI 模型进行精确数值计算

**使用示例：**
```bash
# 启动服务器
uvx mcp-server-calculator

# 服务器会等待输入，可以测试计算功能
# 输入数学表达式，如：2 + 2 * 3
```

### 2. 文件系统服务器 (Filesystem)

**安装命令：**
```bash
# 基本用法（限制访问 /tmp 目录）
uvx @modelcontextprotocol/server-filesystem

# 指定允许访问的目录
uvx @modelcontextprotocol/server-filesystem /path/to/allowed/directory

# 多个目录
uvx @modelcontextprotocol/server-filesystem /path1 /path2 /path3
```

**功能：**
- 安全的文件系统操作
- 文件增删改查
- 目录遍历
- 可通过参数限制访问范围

**使用示例：**
```bash
# 限制访问用户文档目录
uvx @modelcontextprotocol/server-filesystem ~/Documents

# 限制访问当前项目目录
uvx @modelcontextprotocol/server-filesystem ./project
```

### 3. 其他有用的 MCP 服务器

#### 时间服务器
```bash
uvx mcp-server-time
```
获取当前时间、时区信息等

#### Fetch 服务器 (网页内容获取)
```bash
uvx mcp-server-fetch
```
获取网页内容、API响应等

#### PostgreSQL 数据库服务器
```bash
uvx mcp-server-postgres 'postgresql://user:password@localhost/dbname'
```
连接和查询 PostgreSQL 数据库

#### Git 服务器
```bash
uvx @modelcontextprotocol/server-git /path/to/repo
```
Git 仓库操作

#### SQLite 数据库服务器
```bash
uvx mcp-server-sqlite /path/to/database.db
```
SQLite 数据库操作

## 在 Python 项目中使用

### 示例 1：基本用法

```python
import asyncio
import subprocess
from pathlib import Path

class MCPServerManager:
    def __init__(self):
        self.processes = {}
    
    async def start_calculator(self):
        """启动计算器服务器"""
        process = await asyncio.create_subprocess_exec(
            "uvx", "mcp-server-calculator",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.processes['calculator'] = process
        print("✅ 计算器服务器已启动")
    
    async def start_filesystem(self, allowed_dirs=["/tmp"]):
        """启动文件系统服务器"""
        cmd = ["uvx", "@modelcontextprotocol/server-filesystem"] + allowed_dirs
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.processes['filesystem'] = process
        print("✅ 文件系统服务器已启动")
    
    async def stop_all(self):
        """停止所有服务器"""
        for name, process in self.processes.items():
            process.terminate()
            await process.wait()
            print(f"✅ {name} 服务器已停止")

# 使用示例
async def main():
    manager = MCPServerManager()
    
    try:
        await manager.start_calculator()
        await manager.start_filesystem(["/tmp", "./data"])
        
        # 你的应用逻辑...
        await asyncio.sleep(10)
        
    finally:
        await manager.stop_all()

if __name__ == "__main__":
    asyncio.run(main())
```

### 示例 2：与你的现有代码集成

```python
# 在你现有的 mcp_agent_framework_server.py 中添加
import subprocess

def start_mcp_server_with_uvx(server_package, *args):
    """使用 uvx 启动 MCP 服务器"""
    cmd = ["uvx", server_package] + list(args)
    
    print(f"启动 MCP 服务器: {' '.join(cmd)}")
    
    # 返回进程对象，可以进一步处理
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process

# 使用示例
if __name__ == "__main__":
    # 启动计算器服务器
    calc_process = start_mcp_server_with_uvx("mcp-server-calculator")
    
    # 启动文件系统服务器，限制访问当前目录
    fs_process = start_mcp_server_with_uvx(
        "@modelcontextprotocol/server-filesystem", 
        "./"
    )
    
    try:
        # 运行你的主程序
        # ... 你的应用代码 ...
        pass
    finally:
        # 清理资源
        calc_process.terminate()
        fs_process.terminate()
```

## 最佳实践

### 1. 安全考虑

- **文件系统服务器**：始终限制可访问的目录
- **数据库服务器**：使用最小权限原则
- **网络服务器**：确保网络安全配置

### 2. 性能优化

- **进程管理**：正确启动和停止服务器进程
- **资源清理**：确保在应用退出时清理所有服务器
- **错误处理**：实现适当的错误处理和重试机制

### 3. 调试技巧

```bash
# 查看服务器详细输出
uvx mcp-server-calculator --verbose

# 查看错误信息
uvx mcp-server-filesystem 2>&1 | tee mcp.log

# 测试连接
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0.0"}}}' | uvx mcp-server-calculator
```

## 故障排除

### 常见问题

1. **uvx 命令找不到**
   ```bash
   # 确保 uv 已安装并添加到 PATH
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. **服务器启动失败**
   ```bash
   # 检查 Python 环境
   python --version
   uv --version
   
   # 手动安装依赖
   uv pip install mcp
   ```

3. **权限错误**
   ```bash
   # 确保有足够权限访问指定目录
   chmod 755 /path/to/directory
   ```

### 调试命令

```bash
# 检查 uvx 是否正确安装
uvx --version

# 列出已安装的工具
uv tool list

# 卸载工具
uv tool uninstall mcp-server-calculator

# 更新工具
uv tool update-shell
```

## 相关资源

- [MCP 官方文档](https://modelcontextprotocol.io/)
- [MCP 服务器列表](https://github.com/modelcontextprotocol/servers)
- [uvx 文档](https://docs.astral.sh/uv/guides/tools/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)

## 总结

使用 `uvx` 安装 MCP 服务器非常简单和高效。通过这些服务器，你可以为 AI 模型提供强大的工具能力，包括数学计算、文件系统操作、数据库访问等。记住在使用时注意安全配置，并根据你的具体需求选择合适的服务器。