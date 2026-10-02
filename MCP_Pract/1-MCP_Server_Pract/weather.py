from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")


@mcp.tool()
def get_weather(location: str) -> str:
    """Get the weather for a given location."""
    # Placeholder implementation - replace with actual weather API call
    return f"The weather in {location} is sunny."


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
