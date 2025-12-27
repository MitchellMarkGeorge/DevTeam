from .base import BaseLLMClient, LLMClientConfig
from .clients.anthropic import AnthropicClient
from .clients.gemini import GeminiClient
from .clients.openai import OpenAIClient
from .llm_models import (
    ANTHROPIC_MODELS,
    GEMINI_MODELS,
    MODELS,
    OPENAI_MODELS,
    ModelPricing,
    ModelProvider,
    calculate_usage_cost,
    get_default_models_for_agents,
    get_model_by_name_or_snapshot,
    is_reasoning_model,
    validate_model,
)


def create_llm_client(
    config: LLMClientConfig
) -> BaseLLMClient:
    match config.provider:
        case ModelProvider.GEMINI:
            return GeminiClient(config)
        case ModelProvider.ANTHROPIC:
            return AnthropicClient(config)
        case ModelProvider.OPENAI:
            return OpenAIClient(config)
            
__all__ = [
    "GeminiClient",
    "AnthropicClient",
    "OpenAIClient",
    "BaseLLMClient",
    "ModelProvider",
    "ModelPricing",
    "MODELS",
    "ANTHROPIC_MODELS",
    "GEMINI_MODELS",
    "OPENAI_MODELS",
    "validate_model",
    "get_default_models_for_agents",
    "get_model_by_name_or_snapshot",
    "is_reasoning_model",
    "calculate_usage_cost",
]
