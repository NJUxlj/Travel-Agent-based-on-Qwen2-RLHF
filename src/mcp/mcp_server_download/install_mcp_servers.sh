#!/bin/bash

# MCP 服务器安装和使用演示脚本
# 作者: Claude Code Assistant

echo "🔧 MCP 服务器安装和使用演示"
echo "================================"

# 检查 uv 是否安装
if ! command -v uv &> /dev/null; then
    echo "❌ uv 未安装，开始安装..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # 添加到 PATH
    export PATH="$HOME/.local/bin:$PATH"
    echo "✅ uv 安装完成"
else
    echo "✅ uv 已安装"
fi

# 检查 uvx 是否可用
if ! command -v uvx &> /dev/null; then
    echo "❌ uvx 不可用，请重新启动终端或手动添加到 PATH"
    echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
    exit 1
else
    echo "✅ uvx 可用"
fi

echo ""
echo "📦 可用的 MCP 服务器包："
echo "================================"

# 列出一些热门的 MCP 服务器
echo "🔢 计算器服务器:"
echo "  uvx mcp-server-calculator"
echo ""

echo "🗂️  文件系统服务器:"
echo "  uvx @modelcontextprotocol/server-filesystem"
echo "  uvx @modelcontextprotocol/server-filesystem /path/to/allowed/directory"
echo ""

echo "⏰ 时间服务器:"
echo "  uvx mcp-server-time"
echo ""

echo "🔍 搜索服务器:"
echo "  uvx mcp-server-search"
echo ""

echo "🌐 fetch 服务器 (网页内容获取):"
echo "  uvx mcp-server-fetch"
echo ""

echo "💾 PostgreSQL 数据库服务器:"
echo "  uvx mcp-server-postgres 'postgresql://user:password@localhost/dbname'"
echo ""

echo "🧪 测试安装（按 Ctrl+C 停止）..."
echo "================================"

# 函数：测试服务器启动
test_server() {
    local server_name="$1"
    local server_command="$2"
    
    echo ""
    echo "🧪 测试 $server_name..."
    echo "命令: $server_command"
    echo "按 Ctrl+C 跳过测试"
    
    # 设置超时
    timeout 10s bash -c "$server_command" 2>/dev/null &
    local pid=$!
    
    # 等待用户中断或超时
    while kill -0 $pid 2>/dev/null; do
        sleep 1
    done
    
    # 如果进程仍在运行，终止它
    kill $pid 2>/dev/null
    wait $pid 2>/dev/null
    
    echo "✅ $server_name 测试完成"
}

# 提供菜单让用户选择
echo ""
echo "请选择要测试的服务器："
echo "1. 计算器服务器 (mcp-server-calculator)"
echo "2. 文件系统服务器 (@modelcontextprotocol/server-filesystem)"
echo "3. 时间服务器 (mcp-server-time)"
echo "4. fetch 服务器 (mcp-server-fetch)"
echo "5. 测试所有服务器"
echo "6. 退出"

read -p "请输入选择 (1-6): " choice

case $choice in
    1)
        test_server "计算器服务器" "uvx mcp-server-calculator"
        ;;
    2)
        echo ""
        read -p "请输入允许访问的目录 (默认: /tmp): " allowed_dir
        allowed_dir=${allowed_dir:-/tmp}
        test_server "文件系统服务器" "uvx @modelcontextprotocol/server-filesystem $allowed_dir"
        ;;
    3)
        test_server "时间服务器" "uvx mcp-server-time"
        ;;
    4)
        test_server "fetch 服务器" "uvx mcp-server-fetch"
        ;;
    5)
        echo "🔄 测试所有服务器（每个10秒）..."
        test_server "计算器服务器" "uvx mcp-server-calculator"
        test_server "文件系统服务器" "uvx @modelcontextprotocol/server-filesystem /tmp"
        test_server "时间服务器" "uvx mcp-server-time"
        test_server "fetch 服务器" "uvx mcp-server-fetch"
        ;;
    6)
        echo "👋 再见！"
        exit 0
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "🎉 测试完成！"
echo ""
echo "💡 提示："
echo "- 这些服务器在后台运行时会监听标准输入输出"
echo "- 你可以将它们集成到你的 AI 应用中作为工具"
echo "- 更多 MCP 服务器请访问: https://github.com/modelcontextprotocol/servers"
echo ""
echo "📚 相关资源："
echo "- MCP 官方文档: https://modelcontextprotocol.io/"
echo "- 服务器列表: https://github.com/modelcontextprotocol/servers"
echo "- uvx 文档: https://docs.astral.sh/uv/guides/tools/"