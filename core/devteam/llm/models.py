from enum import StrEnum
from typing import Any, Literal, Optional, TypedDict, Union

from pydantic import BaseModel

from devteam.agents.types import AgentType

MessageRole = Literal["user", "assistant"]


class StopReason(StrEnum):
    """Unified stop reason enum across all LLM providers."""

    END_TURN = "end_turn"  # Normal completion
    MAX_TOKENS = "max_tokens"  # Hit token limit
    STOP_SEQUENCE = "stop_sequence"  # Hit a stop sequence
    TOOL_USE = "tool_use"  # Model wants to use tools
    CONTENT_FILTER = "content_filter"  # Content was filtered/blocked
    RECITATION = "recitation"  # Recitation detected (Gemini)
    ERROR = "error"  # Error occurred
    NO_CANDIDATES = "no_candidates"  # No response generated
    UNKNOWN = "unknown"  # Unmapped or unknown reason


class Usage(BaseModel):
    # keeping the model for historical purposes
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    # not using a @property as it is possible that the pricing can change over time
    # and a fixed value should be stored potentially in a db for historical purposes
    cost: Optional[float] = None


class ThinkingData(TypedDict):
    thinking: str
    metadata: str  # Metadata for thinking: signature (Anthropic) or id (OpenAI)
    encrypted_content: Optional[str]


class TextMessage(TypedDict):
    role: MessageRole
    type: Literal["text"]
    text: str
    thinking_data: Optional[ThinkingData]  # Optional thinking/reasoning content
    agent: Optional[AgentType]


class ToolUseData(TypedDict):
    tool_name: str
    tool_use_id: str
    # { param: value }
    arguments: dict[str, Any]  # need to make sure it is in this format


class ToolUseMessage(TypedDict):
    role: Literal["assistant"]
    type: Literal["tool_use"]
    call: ToolUseData
    thinking_data: Optional[ThinkingData]  # Optional thinking/reasoning content
    agent: Optional[AgentType]


class ToolUseResultData(TypedDict):
    tool_name: str
    tool_use_id: str
    result: str # result or error string
    error: bool


class ToolUseResultMessage(TypedDict):
    role: Literal["user"]
    type: Literal["tool_use_result"]
    call_result: ToolUseResultData
    agent: Optional[AgentType]


Message = Union[TextMessage, ToolUseMessage, ToolUseResultMessage]

class LLMResponse(BaseModel):
    content: list[Message]
    stop_reason: Optional[StopReason] = None
    usage: Optional[Usage] = None
