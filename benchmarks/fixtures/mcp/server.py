from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fixture")


@mcp.tool()
def echo(value: str) -> str:
    """Return a deterministic MCP fixture value."""
    return value


if __name__ == "__main__":
    mcp.run(transport="stdio")

