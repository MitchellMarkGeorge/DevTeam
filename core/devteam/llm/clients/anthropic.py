from typing import Optional

import anthropic
from anthropic.types import Message as AnthropicMessage
from anthropic.types import MessageParam, ToolParam

from devteam.llm.base import BaseLLMClient, Message
from devteam.llm.llm_models import ModelProvider, calculate_usage_cost
from devteam.llm.models import (
    LLMResponse,
    StopReason,
    TextMessage,
    ToolUseMessage,
    Usage,
)
from devteam.tools import BaseTool


class AnthropicClient(BaseLLMClient[MessageParam, ToolParam, AnthropicMessage]):
    def __init__(self, model: str, api_key: str, reasoning: bool = False):
        self.client = anthropic.AsyncClient(api_key=api_key)
        super().__init__(ModelProvider.ANTHROPIC, model, api_key, reasoning)

    async def send_message(
        self,
        messages: list[Message],
        system_message: Optional[str] = None,
        tools: Optional[list[BaseTool]] = None,  # look at type definition
        max_tokens: int = 16_384,  # think about this (2 ** 14)
        temperature: float = 0.7,
    ) -> LLMResponse:
        converted_tools: list[ToolParam] | None = (
            [self._convert_tool(tool) for tool in tools] if tools else None
        )

        converted_messages = [self._convert_message(message) for message in messages]

        args = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": converted_messages,
            "temperature": temperature,
            "tool_choice": {
                "type": "auto",
                "disable_parallel_tool_use": True,  # think about this
            },
        }

        if system_message:
            args["system"] = system_message

        if tools:
            args["tools"] = converted_tools

        message = await self._call_llm_api_with_retry(**args)
        return self._convert_llm_response(message)

    async def _call_llm_api(self, **kwargs) -> AnthropicMessage:
        return await self.client.messages.create(**kwargs)

    def _convert_tool(self, tool: BaseTool) -> ToolParam:
        schema = tool.schema
        properties = {}
        required = []

        for param in schema.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum

            if param.default:
                properties[param.name]["default"] = param.default

            if param.required:
                required.append(param.name)

        return {
            "name": schema.name,
            "description": schema.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def _convert_message(self, message: Message) -> MessageParam:
        match message["type"]:
            case "text":
                return {
                    "role": message["role"],
                    "content": message["text"],
                }
            case "tool_use":
                return {
                    "role": message["role"],
                    "content": [
                        {
                            "type": "tool_use",
                            "id": message["call"]["tool_use_id"],
                            "name": message["call"]["tool_name"],
                            "input": message["call"]["arguments"],
                        }
                    ],
                }
            case "tool_use_result":
                return {
                    "role": message["role"],
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message["call_result"]["tool_use_id"],
                            "content": message["call_result"]["result"],
                            "is_error": message["call_result"]["error"],
                        }
                    ],
                }

    def _convert_llm_response(self, response: AnthropicMessage) -> LLMResponse:
        content_blocks: list[Message] = []
        for block in response.content:
            if block.type == "text":
                text_message: TextMessage = {
                    "role": "assistant",
                    "text": block.text,
                    "type": "text",
                }
                content_blocks.append(text_message)
            elif block.type == "tool_use":
                tool_use_message: ToolUseMessage = {
                    "type": "tool_use",
                    "role": "assistant",
                    "call": {
                        "tool_name": block.name,
                        "tool_use_id": block.id,
                        "arguments": block.input,
                    },
                }
                content_blocks.append(tool_use_message)

        # Map Anthropic stop reasons to our enum
        stop_reason_map = {
            "end_turn": StopReason.END_TURN,
            "max_tokens": StopReason.MAX_TOKENS,
            "stop_sequence": StopReason.STOP_SEQUENCE,
            "tool_use": StopReason.TOOL_USE,
        }
        stop_reason = (
            stop_reason_map.get(response.stop_reason, StopReason.UNKNOWN)
            if response.stop_reason
            else None
        )

        # Extract token usage and calculate cost
        usage = None
        if response.usage:
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = calculate_usage_cost(
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            usage = Usage(
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
            )

        return LLMResponse(
            content=content_blocks,
            stop_reason=stop_reason,
            usage=usage,
        )

    async def close(self):
        await self.client.close()
