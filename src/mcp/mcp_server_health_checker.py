#!/usr/bin/env python3
"""
简单的 MCP 工具检查器
检查 MCP 服务器的工具列表和基本功能
"""
import subprocess
import sys
import json

def check_mcp_tools(server_path, server_name):
    """检查 MCP 服务器的工具列表"""
    print(f"\n=== Checking {server_name} ===")
    
    try:
        # 发送 initialize 请求
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        # 发送 tools/list 请求
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        # 组合请求
        requests = [init_request, tools_request]
        
        # 启动服务器进程
        process = subprocess.Popen(
            [sys.executable, server_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 发送请求
        for req in requests:
            request_line = json.dumps(req) + "\n"
            process.stdin.write(request_line)
            process.stdin.flush()
            
        # 等待短暂时间让服务器响应
        import time
        time.sleep(0.5)
        
        # 读取响应
        stdout_lines = []
        while process.poll() is None:
            try:
                line = process.stdout.readline()
                if line:
                    stdout_lines.append(line)
                else:
                    break
            except:
                break
        
        # 终止进程
        process.terminate()
        process.wait(timeout=1)
        
        # 解析响应
        print(f"服务器输出:")
        for line in stdout_lines:
            print(f"  {line.strip()}")
            
        stderr_output = process.stderr.read()
        if stderr_output:
            print(f"错误输出:")
            print(f"  {stderr_output}")
            
    except Exception as e:
        print(f"检查 {server_name} 时出错: {e}")

def main():
    """主函数"""
    servers = [
        ("....", "Weather Server"),
        ("....", "Documentation Server"),
        ("....", "Real-time Data Server")
    ]
    
    print("MCP 服务器工具检查器")
    print("=" * 50)
    
    for server_path, server_name in servers:
        check_mcp_tools(server_path, server_name)
    
    print("\n" + "=" * 50)
    print("检查完成!")

if __name__ == "__main__":
    main()