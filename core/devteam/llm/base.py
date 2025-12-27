from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from devteam.llm.llm_models import ModelProvider, is_reasoning_model, validate_model
from devteam.llm.models import LLMResponse, Message
from devteam.tools import BaseTool
from devteam.utils import exponential_backoff_retry

TMessage = TypeVar("TMessage")
TTool = TypeVar("TTool")
TResponse = TypeVar("TResponse")


@dataclass
class LLMClientConfig:
    provider: ModelProvider
    api_key: str
    model: str
    reasoning_enabled: bool = False


class BaseLLMClient(ABC, Generic[TMessage, TTool, TResponse]):
    def __init__(self, config: LLMClientConfig):
        (is_valid, error_message) = self._validate_model(
            config.provider, config.model, config.reasoning_enabled
        )
        if not is_valid and error_message:
            raise ValueError(error_message)
        self.model = config.model
        self.api_key = config.api_key
        self.reasoning_enabled = config.reasoning_enabled

    def _validate_model(
        self, provider: ModelProvider, model: str, reasoning: bool
    ) -> tuple[bool, str | None]:
        is_model_valid = validate_model(provider, model)

        if not is_model_valid:
            return False, f"Invalid {provider.title()} model {model}"

        if self.reasoning_enabled and not is_reasoning_model(model):
            return False, f"Model {model} does not support reasoning"

        return (
            True,
            None,
        )

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        system_message: Optional[str] = None,
        tools: Optional[list[BaseTool]] = None,  # look at type definition
        max_tokens: int = 32_768,  # think about this (2 ** 15)
        temperature: float = 0.7,
    ) -> LLMResponse:
        # TODO: should the agent type be passed in for the due to the response messages
        # technically for streaming, this won't be needed as the messages will be put together
        raise NotImplementedError()

    @abstractmethod
    async def _call_llm_api(self, **kwargs) -> TResponse:
        """
        Makes the actual API call to the LLM provider.
        This method should only contain the API call logic and will be wrapped with retry logic.

        Args:
            **kwargs: Provider-specific arguments for the API call

        Returns:
            TResponse: The raw response from the provider
        """
        raise NotImplementedError()

    # disabling this for now
    # @exponential_backoff_retry(retries=5, delay=1.0)
    async def _call_llm_api_with_retry(self, **kwargs) -> TResponse:
        """
        Wrapper that adds exponential backoff retry logic to the API call.
        Child classes should call this method instead of _call_llm_api directly.

        Args:
            **kwargs: Provider-specific arguments for the API call

        Returns:
            TResponse: The raw response from the provider
        """
        return await self._call_llm_api(**kwargs)

    @abstractmethod
    def _convert_message[T](self, message: Message) -> TMessage:
        # convert the given message to the required format based on the provider
        raise NotImplementedError()

    @abstractmethod
    def _convert_tool[T](self, tool: BaseTool) -> TTool:
        # convert the given tool to the required format based on the provider
        raise NotImplementedError()

    @abstractmethod
    def _convert_llm_response(self, response: TResponse) -> LLMResponse:
        # convert the given response from the provider to a unified format
        # Shouldn't the LLMRResponse have Messages instead?
        raise NotImplementedError()

    @abstractmethod
    async def close(self):
        raise NotImplementedError()
