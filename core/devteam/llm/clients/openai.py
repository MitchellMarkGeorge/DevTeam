import json
from typing import Any, Optional

from openai import AsyncClient
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    Response,
    ResponseFunctionToolCallParam,
    ResponseInputItemParam,
)
from openai.types.responses.response_input_item_param import FunctionCallOutput

from devteam.llm.base import BaseLLMClient
from devteam.llm.llm_models import ModelProvider, calculate_usage_cost
from devteam.llm.models import (
    LLMResponse,
    Message,
    TextMessage,
    ToolUseMessage,
    StopReason,
    Usage,
)
from devteam.tools import BaseTool


class OpenAIClient(BaseLLMClient[ResponseInputItemParam, FunctionToolParam, Response]):
    def __init__(self, model: str, api_key: str, reasoning: bool = False):
        self.client = AsyncClient(api_key=api_key)
        super().__init__(ModelProvider.OPENAI, model, api_key, reasoning)

    async def send_message(
        self,
        messages: list[Message],
        system_message: Optional[str] = None,
        tools: Optional[list[BaseTool]] = None,  # look at type definition
        max_tokens: int = 16_384,  # think about this (2 ** 14)
        temperature: float = 0.7,
    ) -> LLMResponse:
        # Implement the send_message method here
        converted_messages = []

        for message in messages:
            converted_messages.append(self._convert_message(message))

        args = {
            "model": self.model,
            "input": converted_messages,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            # think about this
            "parallel_tool_calls": False,  # need to handle tool calls individually especially for tool calls that require user approval
            "store": False,  # we want to manage context ourselves
        }

        if system_message:
            args["instructions"] = system_message

        if tools:
            args["tools"] = [self._convert_tool(tool) for tool in tools]

        response: Response = await self._call_llm_api_with_retry(**args)
        return self._convert_llm_response(response)

    async def _call_llm_api(self, **kwargs) -> Response:
        return self.client.responses.create(**kwargs)

    def _convert_tool(self, tool: BaseTool) -> FunctionToolParam:
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
            "type": "function",
            "name": schema.name,
            "description": schema.description,
            "strict": True,  # think about this
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def _convert_message(
        self, message: Message
    ) -> ResponseInputItemParam:  # not sure of correct output type here
        match message["type"]:
            case "text":
                text_message: EasyInputMessageParam = {
                    "role": message["role"],
                    "content": message["text"],
                }
                return text_message
            case "tool_use":
                tool_use_message: ResponseFunctionToolCallParam = {
                    "type": "function_call",
                    "name": message["call"]["tool_name"],
                    "call_id": message["call"]["tool_use_id"],
                    "arguments": json.dumps(message["call"]["arguments"]),
                }
                return tool_use_message
            case "tool_use_result":
                tool_use_result_message: FunctionCallOutput = {
                    "type": "function_call_output",
                    "call_id": message["call_result"]["tool_use_id"],
                    "output": message["call_result"]["result"],
                }
                return tool_use_result_message


    def _convert_llm_response(self, response: Response) -> LLMResponse:
        content_block: list[Message] = []
        for block in response.output:
            if block.type == "message":
                # should it go through the different content array options as different blocks are merge them into one?
                # content_block.append(
                #     LLMResponseTextContentBlock(type="text", text=)
                # )
                for content in block.content:
                    # Should I handle refusal?
                    if content.type == "output_text":
                        text_message: TextMessage = {
                            "role": "assistant",
                            "text": content.text,
                            "type": "text",
                        }
                        content_block.append(
                            text_message
                        )
            elif block.type == "function_call":
                arguments: dict[str, Any] = json.loads(block.arguments)
                tool_use_message: ToolUseMessage = {
                    "type": "tool_use",
                    "role": "assistant",
                    "call": {
                        "tool_name": block.name,
                        "tool_use_id": block.call_id,
                        "arguments": arguments,
                    },
                }
                content_block.append(
                    tool_use_message
                )

        # Map OpenAI status to our enum
        # OpenAI uses "completed", "incomplete", "failed", etc.
        stop_reason_map = {
            "completed": StopReason.END_TURN,
            "incomplete": StopReason.MAX_TOKENS,  # Usually means hit token limit
            "failed": StopReason.ERROR,
        }
        stop_reason = (
            stop_reason_map.get(response.status, StopReason.UNKNOWN)
            if response.status
            else None
        )

        # Extract token usage and calculate cost
        usage = None
        if response.usage:
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = calculate_usage_cost(self.model, input_tokens, output_tokens)
            usage = Usage(
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
            )

        return LLMResponse(
            content=content_block,
            stop_reason=stop_reason,
            usage=usage,
        )

    async def close(self):
        return await self.client.close()
