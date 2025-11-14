#!/usr/bin/env python3
"""
SSE MCP Server - 提供文档查询功能
"""
import sys
import logging
from typing import Any
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("docs-server")

# Mock documentation data
DOCS_DATA = {
    "python": {
        "title": "Python Documentation",
        "description": "Python is a high-level programming language",
        "functions": ["print()", "len()", "range()", "str()", "int()"]
    },
    "javascript": {
        "title": "JavaScript Documentation", 
        "description": "JavaScript is a programming language that runs in web browsers",
        "functions": ["console.log()", "Array.map()", "JSON.parse()", "setTimeout()"]
    },
    "mcp": {
        "title": "MCP (Model Context Protocol)",
        "description": "MCP is a protocol for connecting AI assistants to external systems and data sources",
        "features": ["stdio transport", "websocket transport", "streamable HTTP transport"]
    }
}

@mcp.tool()
async def search_documentation(topic: str) -> str:
    """Search for documentation about a specific topic.
    
    Args:
        topic: The topic to search for (e.g., "python", "javascript", "mcp")
    """
    topic = topic.lower()
    
    if topic not in DOCS_DATA:
        available_topics = ", ".join(DOCS_DATA.keys())
        return f"Documentation not found for '{topic}'. Available topics: {available_topics}"
    
    data = DOCS_DATA[topic]
    result = f"""
Topic: {data['title']}
Description: {data['description']}
Functions/Features: {', '.join(data['functions'])}
"""
    return result

@mcp.tool()
async def list_all_topics() -> str:
    """List all available documentation topics."""
    topics = list(DOCS_DATA.keys())
    return f"Available documentation topics: {', '.join(topics)}"

def main():
    # Initialize and run the server
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()