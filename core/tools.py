from abc import ABC, abstractmethod
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType


class Tool(ABC):
    """Base class for agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON schema for tool parameters."""
        ...

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """Execute the tool with given arguments.

        Returns:
            Result as string for LLM consumption
        """
        ...

    @property
    def schema(self) -> dict:
        """Get OpenAI-compatible tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolsManager:
    def __init__(self, autoload: bool = True, tools_dir: Path | None = None):
        self._tools: dict[str, Tool] = {}
        if autoload:
            self.autoload(tools_dir=tools_dir)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name '{tool.name}'")
        self._tools[tool.name] = tool

    def autoload(self, tools_dir: Path | None = None) -> None:
        base_dir = Path(__file__).resolve().parents[1]
        directory = tools_dir or (base_dir / "tools")

        for file_path in sorted(directory.glob("*.py")):
            if file_path.stem.startswith("_"):
                continue
            module = self._load_module(file_path)
            self._register_module_tools(module)

    def _load_module(self, file_path: Path) -> ModuleType:
        module_name = f"oclaw_tool_{file_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load tool module: {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _register_module_tools(self, module: ModuleType) -> None:
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj is Tool or not issubclass(obj, Tool):
                continue
            if obj.__module__ != module.__name__:
                continue
            self.register(obj())

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_schemas(self) -> list[dict]:
        return [tool.schema for tool in self._tools.values()]

    async def execute(self, name: str, args: dict) -> str:
        tool = self.get(name)
        if not tool:
            return f"Error: Unknown tool '{name}'"

        try:
            return await tool.execute(**args)
        except Exception as e:
            return f"Error: {str(e)}"
