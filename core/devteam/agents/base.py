from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from devteam.agents.types import AgentType
from devteam.llm import create_llm_client
from devteam.llm.base import LLMClientConfig
from devteam.llm.models import Message, TextMessage, ToolUseData, ToolUseResultMessage
from devteam.state.models.context import Context
from devteam.tools import BaseTool, ToolResult

class ToolNotFoundException(Exception):
    pass


class BaseAgent(ABC):
    def __init__(
        self,
        type: AgentType,
        # tools: dict[str, BaseTool],
        tools: list[BaseTool],
        model_config: LLMClientConfig,
        fallback_model_config: Optional[LLMClientConfig],
        # we don't want the agent to just go on forever
        max_turns: int = 10,
        # think about this
        # max_tool_calls: int = 5
    ):
        self.type = type
        self.llm_client = create_llm_client(model_config)

        if fallback_model_config is not None:
            self.fallback_llm_client = create_llm_client(fallback_model_config)
        else:
            self.fallback_llm_client = None

        self.tool_map: dict[str, BaseTool] = {tool.schema.name: tool for tool in tools}
        self.tools = tools
        self.max_turns = max_turns
        
        # these are all the messages for the agent
        # the idea is that you can also have "internal" messages here
        self.messages: list[Message] = []

    @property
    @abstractmethod
    def system_message(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    async def invoke(
        self, context: Context, prompt: Optional[str] = None
    ) -> AsyncIterator[Message]:
        if prompt is not None:
            prompt_message: TextMessage = {
                "type": "text",
                "role": "user",
                "text": prompt,
                "agent": self.type,
                "thinking_data": None,
            }
            # context.add_message(message)
            self.messages.append(prompt_message)

        for _ in range(self.max_turns):
            # TODO: the agent should be aware of the number of turns left
            # might have to be part of the system message
            try:
                response = await self.llm_client.complete(
                    system_message=self.system_message,
                    messages=self.messages,
                    tools=self.tools,
                )
                # TODO: try and specity type of exception
            except Exception as e:
                # error handling

                # try fallback client
                if self.fallback_llm_client is not None:
                    try:
                        response = await self.fallback_llm_client.complete(
                            system_message=self.system_message,
                            messages=self.messages,
                            tools=self.tools,
                        )
                    except Exception as e:
                        raise e

                else:
                    raise e

            # could also try an use the stop_reason
            has_tool_use = False

            for message in response.content:
                # make sure to set the appropriate agent type
                message["agent"] = self.type

                self.messages.append(message)
                yield message

                if message["type"] == "tool_use":
                    has_tool_use = True
                    result = await self._handle_tool_use(message["call"])
                    
                    tool_use_result_message: ToolUseResultMessage = {
                        "role": "user",
                        "type": "tool_use_result",
                        "agent": self.type,
                        "call_result": { # come back to this
                            "result": str(result.data),
                            "error": bool(result.error),
                            "tool_use_id": message["call"]["tool_use_id"],
                            "tool_name": message["call"]["tool_name"],
                        }
                    }

                    self.messages.append(tool_use_result_message)
                    yield tool_use_result_message

                # assume that the agent is done if there are no tool uses
                if not has_tool_use: # should we check the stop reason?
                    break

    async def _handle_tool_use(self, tool_use: ToolUseData) -> ToolResult:
        tool = self.tool_map.get(tool_use["tool_name"])

        if not tool:
            raise ToolNotFoundException(f"Tool {tool_use['tool_name']} not found")

        result = await tool.execute(**tool_use["arguments"])
        return result
