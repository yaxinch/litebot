"""Tool registry for dynamic tool management."""

from dataclasses import dataclass
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.security.tool_policy import ToolErrorClassifier, ToolErrorInfo, ToolOutcome


@dataclass(slots=True)
class ToolExecutionResult:
    content: Any
    status: str = "ok"
    stage: str | None = None
    error: str | None = None
    outcome: ToolOutcome = ToolOutcome.SUCCESS
    progress: dict[str, Any] | None = None
    error_info: ToolErrorInfo | None = None
    normalized_arguments: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status == "error" and self.outcome is ToolOutcome.SUCCESS:
            self.outcome = ToolOutcome.FAILED
        if self.status == "error" and self.error_info is None:
            self.error_info = ToolErrorClassifier.classify(
                self.error or self.content, stage=self.stage or "result"
            )

    @classmethod
    def partial(cls, content: Any, **progress: Any) -> "ToolExecutionResult":
        return cls(content=content, outcome=ToolOutcome.PARTIAL, progress=progress or None)


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
            return ToolExecutionResult(
                content, "error", "lookup", content, ToolOutcome.FAILED,
                error_info=ToolErrorClassifier.classify(content, stage="lookup"),
            )

        try:
            # Attempt to cast parameters to match schema types
            params = tool.cast_params(params)

            # Validate parameters
            errors = tool.validate_params(params)
            if errors:
                content = f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors) + hint
                return ToolExecutionResult(
                content, "error", "validation", content, ToolOutcome.FAILED,
                error_info=ToolErrorClassifier.classify(content, stage="validation"),
                normalized_arguments=params,
            )
            result = await tool.execute(**params)
            if isinstance(result, ToolExecutionResult):
                if result.normalized_arguments is None:
                    result.normalized_arguments = params
                return result
            is_error = isinstance(result, str) and (
                result.startswith("Error")
                or "timed out" in result.lower()
                or result.startswith("(MCP tool call was cancelled")
            )
            if is_error:
                content = result + hint
                return ToolExecutionResult(
                    content, "error", "result", content, ToolOutcome.FAILED,
                    error_info=ToolErrorClassifier.classify(result, stage="result"),
                    normalized_arguments=params,
                )
            return ToolExecutionResult(result, normalized_arguments=params)
        except Exception as e:
            content = f"Error executing {name}: {str(e)}" + hint
            return ToolExecutionResult(
                content, "error", "execution", str(e), ToolOutcome.FAILED,
                error_info=ToolErrorClassifier.classify(e, stage="execution"),
                normalized_arguments=params,
            )

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
