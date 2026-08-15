"""Tool registry for dynamic tool management."""

from dataclasses import dataclass
from typing import Any

from nanobot.agent.tools.base import Tool


@dataclass(slots=True)
class ToolExecutionResult:
    content: Any
    status: str = "ok"
    stage: str | None = None
    error: str | None = None


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> Any:
        """Execute a tool by name with given parameters."""
        return (await self.execute_detailed(name, params)).content

    async def execute_detailed(self, name: str, params: dict[str, Any]) -> ToolExecutionResult:
        """Execute a tool while preserving error classification for lifecycle observers."""
        hint = "\n\n[Analyze the error above and try a different approach.]"

        tool = self._tools.get(name)
        if not tool:
            content = f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}"
            return ToolExecutionResult(content, "error", "lookup", content)

        try:
            # Attempt to cast parameters to match schema types
            params = tool.cast_params(params)

            # Validate parameters
            errors = tool.validate_params(params)
            if errors:
                content = f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors) + hint
                return ToolExecutionResult(content, "error", "validation", content)
            result = await tool.execute(**params)
            if isinstance(result, str) and result.startswith("Error"):
                content = result + hint
                return ToolExecutionResult(content, "error", "result", content)
            return ToolExecutionResult(result)
        except Exception as e:
            content = f"Error executing {name}: {str(e)}" + hint
            return ToolExecutionResult(content, "error", "execution", str(e))

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# Runner uses the detailed fast path only while the public execution method has
# not been overridden. This preserves subclass and monkeypatch compatibility.
DEFAULT_TOOL_EXECUTE = ToolRegistry.execute
