from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, Optional, TypeVar


class ToolParameterType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter[T]:
    name: str
    description: str
    type: ToolParameterType
    required: bool = True
    enum: Optional[list[T]] = None
    default: T | None = None


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: list[ToolParameter]


T = TypeVar("T")


@dataclass
class ToolResult(Generic[T]):
    success: bool
    data: T
    error: Optional[str] = None
    duration_ms: int = 0


class BaseTool(ABC):
    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        """Return the tool definition schema."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with the given arguments."""
        pass

    def validate_args(self, kwargs: dict):
        pass
