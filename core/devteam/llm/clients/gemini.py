from typing import Optional
import json

from google.genai import Client
from google.genai.types import (
    ContentDict,
    FunctionDeclarationDict,
    GenerateContentConfigOrDict,
    GenerateContentResponse,
    SchemaDict,
    ToolDict,
    Type,
)

from devteam.llm.base import BaseLLMClient
from devteam.llm.llm_models import ModelProvider, calculate_usage_cost
from devteam.llm.models import (
    LLMResponse,
    Message,
    StopReason,
    TextMessage,
    ToolUseMessage,
    Usage,
)
from devteam.tools import BaseTool, ToolParameterType


class GeminiClient(BaseLLMClient[ContentDict, ToolDict, GenerateContentResponse]):
    def __init__(self, model: str, api_key: str, reasoning: bool = False):
        self.client = Client(api_key=api_key).aio
        super().__init__(ModelProvider.GEMINI, model, api_key, reasoning)

    async def send_message(
        self,
        messages: list[Message],
        system_message: Optional[str] = None,
        tools: Optional[list[BaseTool]] = None,  
        max_tokens: int = 16_384,  # think about this (2 ** 14)
        temperature: float = 0.7,
    ) -> LLMResponse:
        converted_messages = [self._convert_message(message) for message in messages]

        config: GenerateContentConfigOrDict = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools:
            config["tools"] = [self._convert_tool(tool) for tool in tools]

        if system_message:
            config["system_instruction"] = system_message

        args = {
            "model": self.model,
            "contents": converted_messages,
            "config": config,
        }

        response = await self._call_llm_api_with_retry(**args)

        return self._convert_llm_response(response)

    async def _call_llm_api(self, **kwargs) -> GenerateContentResponse:
        return await self.client.models.generate_content(**kwargs)

    def _convert_message(self, message: Message) -> ContentDict:
        # TODO: handle tool calls and other message possibilities
        match message["type"]:
            case "text":
                role = message["role"] if message["role"] == "user" else "model"
                return {
                    "role": role,
                    "parts": [{"text": message["text"]}],
                }

            case "tool_use":
                tool_call = message["call"]
                return {
                    "role": "model",
                    "parts": [
                        {
                            "function_call": {
                                "id": tool_call.get("tool_use_id", None),
                                "name": tool_call["tool_name"],
                                "args": tool_call["arguments"],
                            }
                        }
                    ],
                }
            
            case "tool_use_result":
                tool_call_result = message["call_result"]
                return {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "id": tool_call_result.get("tool_use_id", None),
                                "name": tool_call_result["tool_name"],
                                "response": json.loads(tool_call_result["result"])
                            }
                        }
                    ],
                }

    def _convert_tool(self, tool: BaseTool) -> ToolDict:
        schema = tool.schema

        properties: dict[str, SchemaDict] = {}
        required: list[str] = []

        def _convert_parameter_type(param_type: ToolParameterType) -> Type:
            match param_type:
                case ToolParameterType.STRING:
                    return Type.STRING
                case ToolParameterType.INTEGER:
                    return Type.INTEGER
                case ToolParameterType.NUMBER:
                    return Type.NUMBER
                case ToolParameterType.BOOLEAN:
                    return Type.BOOLEAN
                case ToolParameterType.ARRAY:
                    return Type.ARRAY
                case ToolParameterType.OBJECT:
                    return Type.OBJECT

        for param in schema.parameters:
            tool_schema: SchemaDict = {
                "type": _convert_parameter_type(param.type),
                "description": param.description,
            }
            if param.enum:
                tool_schema["enum"] = param.enum

            if param.default:
                tool_schema["default"] = param.default

            if param.required:
                required.append(param.name)

            properties[param.name] = tool_schema

        function_declaration: FunctionDeclarationDict = {
            "name": schema.name,
            "description": schema.description,
            "parameters": {
                "type": Type.OBJECT,
                "properties": properties,
                "required": required,
            },
        }
        return {"function_declarations": [function_declaration]}

    def _convert_llm_response(self, response: GenerateContentResponse) -> LLMResponse:
        best_candidate = response.candidates[0] if response.candidates else None

        # This happens when the API returns no candidates (e.g., content filtered, error, etc.)
        EMPTY_LLM_RESPONSE = LLMResponse(
            content=[], stop_reason=StopReason.NO_CANDIDATES
        )

        if (
            not best_candidate
            or not best_candidate.content
            or not best_candidate.content.parts
        ):
            return EMPTY_LLM_RESPONSE

        content_blocks: list[Message] = []

        for part in best_candidate.content.parts:
            if hasattr(part, "text") and part.text is not None:
                text_message: TextMessage = {
                    "role": "assistant",
                    "type": "text",
                    "text": part.text
                }
                content_blocks.append(text_message)
            elif (
                hasattr(part, "function_call")
                and part.function_call
                and part.function_call.name is not None
                and part.function_call.args
                and part.function_call.id is not None # think about this one
            ):
                part.function_call.id
                tool_use_message: ToolUseMessage = {
                    "type": "tool_use",
                    "role": "assistant",
                    "call": {
                        "tool_name": part.function_call.name,
                        "tool_use_id": part.function_call.id,
                        "arguments": part.function_call.args,
                    },
                }
                content_blocks.append(
                    tool_use_message
                )

        # Map Gemini finish_reason to our enum
        # Gemini uses: "STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "OTHER", etc.
        finish_reason = (
            best_candidate.finish_reason
            if hasattr(best_candidate, "finish_reason")
            else None
        )
        stop_reason_map = {
            "STOP": StopReason.END_TURN,
            "MAX_TOKENS": StopReason.MAX_TOKENS,
            "SAFETY": StopReason.CONTENT_FILTER,
            "RECITATION": StopReason.RECITATION,
            "OTHER": StopReason.UNKNOWN,
        }
        stop_reason = (
            stop_reason_map.get(finish_reason, StopReason.UNKNOWN)
            if finish_reason
            else None
        )

        # Extract token usage and calculate cost
        # Gemini has usage_metadata with prompt_token_count and candidates_token_count
        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = (
                response.usage_metadata.prompt_token_count
                if hasattr(response.usage_metadata, "prompt_token_count")
                else None
            )
            output_tokens = (
                response.usage_metadata.candidates_token_count
                if hasattr(response.usage_metadata, "candidates_token_count")
                else None
            )
            if input_tokens is not None and output_tokens is not None:
                cost = calculate_usage_cost(self.model, input_tokens, output_tokens)
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
        await self.client.aclose()
