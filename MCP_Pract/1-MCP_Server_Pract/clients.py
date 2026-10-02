import asyncio
import os

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()


async def main():

    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["mathserver.py"],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable-http",
            },
        }
    )

    # Get tools from both MCP servers
    tools = await client.get_tools()

    print("Available tools:")
    for tool in tools:
        print(tool.name)

    # OpenRouter model
    model = ChatOpenAI(
        model="prism-ml/ternary-bonsai-2-27b",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        max_tokens=1000,
    )

    # Create agent with MCP tools
    agent = create_agent(
        model=model,
        tools=tools,
    )

    # Ask the agent to use the math MCP server
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is (5 + 3) * 12?"}]}
    )
    for m in response["messages"]:
        m.pretty_print()

    print("\nFinal Response:")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
