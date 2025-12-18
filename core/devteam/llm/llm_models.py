from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple


class ModelPricing(NamedTuple):
    input_price: float  # Price per 1M input tokens
    output_price: float  # Price per 1M output tokens


class ModelProvider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass
class LLMModel:
    name: str  # this is normally the default alias that maps the latest model version
    provider: ModelProvider
    snapshots: list[str]
    pricing: ModelPricing
    context_window_size: int
    has_reasoning: bool = False 

    def get_default_model_id(self) -> str:
        return self.snapshots[0] or self.name


MODELS = {
    # Anthropic Models
    "claude-haiku-4-5": LLMModel(
        provider=ModelProvider.ANTHROPIC,
        name="claude-haiku-4-5",
        snapshots=["claude-haiku-4-5-20251001"],
        pricing=ModelPricing(1.00, 5.00),
        context_window_size=200_000,
    ),
    "claude-sonnet-4-5": LLMModel(
        provider=ModelProvider.ANTHROPIC,
        name="claude-sonnet-4-5",
        snapshots=["claude-sonnet-4-5-20250929"],
        pricing=ModelPricing(3.00, 15.00),
        context_window_size=200_000,
        has_reasoning=True,
    ),
    "claude-opus-4-5": LLMModel(
        provider=ModelProvider.ANTHROPIC,
        name="claude-opus-4-5",
        snapshots=["claude-opus-4-5-20251101"],
        pricing=ModelPricing(5.00, 25.00),
        context_window_size=200_000,
        has_reasoning=True,
    ),
    
    # OpenAI Models
    "gpt-5-mini": LLMModel(
        provider=ModelProvider.OPENAI,
        name="gpt-5-mini",
        snapshots=["gpt-5-mini-2025-08-07"],
        pricing=ModelPricing(0.25, 2.00),
        context_window_size=400_000,
        has_reasoning=True,
    ),
    "gpt-5": LLMModel(
        provider=ModelProvider.OPENAI,
        name="gpt-5",
        snapshots=["gpt-5-2025-08-07"],
        pricing=ModelPricing(1.25, 10.00),
        context_window_size=400_000,
        has_reasoning=True,
    ),
    "gpt-5.1": LLMModel(
        provider=ModelProvider.OPENAI,
        name="gpt-5.1",
        snapshots=["gpt-5.1-2025-11-13"],
        pricing=ModelPricing(1.25, 10.00),
        context_window_size=400_000,
        has_reasoning=True,
    ),
    "gpt-5-pro": LLMModel(
        provider=ModelProvider.OPENAI,
        name="gpt-5-pro",
        snapshots=["gpt-5-pro-2025-10-06"],
        pricing=ModelPricing(15.00, 120.00),
        context_window_size=400_000,
        has_reasoning=True,
    ),
    
    # Gemini Models
    # Disabling reasoning on Gemeni models for now so I can get it right
    # 
    "gemini-2.5-flash-lite": LLMModel(
        provider=ModelProvider.GEMINI,
        name="gemini-2.5-flash-lite",
        snapshots=[],
        pricing=ModelPricing(0.10, 0.40),
        context_window_size=1_048_576,
        # has_reasoning=True,
    ),
    "gemini-2.5-flash": LLMModel(
        provider=ModelProvider.GEMINI,
        name="gemini-2.5-flash",
        snapshots=[],
        pricing=ModelPricing(0.30, 2.50),
        context_window_size=1_048_576,
        # has_reasoning=True,
    ),
    "gemini-2.5-pro": LLMModel(
        provider=ModelProvider.GEMINI,
        name="gemini-2.5-pro",
        snapshots=[],
        pricing=ModelPricing(1.25, 10.00),
        context_window_size=1_048_576,
        # has_reasoning=True,
    ),
}


ANTHROPIC_MODELS = {
    model.name: model
    for model in MODELS.values()
    if model.provider == ModelProvider.ANTHROPIC
}

OPENAI_MODELS = {
    model.name: model
    for model in MODELS.values()
    if model.provider == ModelProvider.OPENAI
}

GEMINI_MODELS = {
    model.name: model
    for model in MODELS.values()
    if model.provider == ModelProvider.GEMINI
}


def get_model_by_name_or_snapshot(name_or_version: str) -> LLMModel | None:
    for model in MODELS.values():
        if model.name == name_or_version or name_or_version in model.snapshots:
            return model
    return None


def validate_model(provider: ModelProvider, model: str) -> bool:
    llm_model = get_model_by_name_or_snapshot(model)
    return llm_model is not None and llm_model.provider == provider
    
def is_reasoning_model(model: str) -> bool:
    llm_model = get_model_by_name_or_snapshot(model)
    return llm_model is not None and llm_model.has_reasoning

def get_default_anthorpic_models_for_agents():
    """Get default Anthropic models for each agent role."""
    return {
        "manager": ANTHROPIC_MODELS["claude-haiku-4-5"].get_default_model_id(),
        "architect": ANTHROPIC_MODELS["claude-sonnet-4-5"].get_default_model_id(),
        "developer": ANTHROPIC_MODELS["claude-sonnet-4-5"].get_default_model_id(),
        "qa": ANTHROPIC_MODELS["claude-sonnet-4-5"].get_default_model_id(),
    }


def get_default_openai_models_for_agents():
    """Get default OpenAI models for each agent role."""
    return {
        "manager": OPENAI_MODELS["gpt-5-mini"].get_default_model_id(),
        "architect": OPENAI_MODELS["gpt-5.1"].get_default_model_id(),
        "developer": OPENAI_MODELS["gpt-5.1"].get_default_model_id(),
        "qa": OPENAI_MODELS["gpt-5.1"].get_default_model_id(),
    }


def get_default_gemini_models_for_agents():
    """Get default Gemini models for each agent role."""
    return {
        "manager": GEMINI_MODELS["gemini-2.5-flash-lite"].get_default_model_id(),
        "architect": GEMINI_MODELS["gemini-2.5-flash"].get_default_model_id(),
        "developer": GEMINI_MODELS["gemini-2.5-flash"].get_default_model_id(),
        "qa": GEMINI_MODELS["gemini-2.5-flash"].get_default_model_id(),
    }


def get_default_models_for_agents(provider: ModelProvider):
    """
    Get default models for each agent role based on provider.

    Args:
        provider: The model provider

    Returns:
        Dictionary mapping agent roles to model names
    """
    match provider:
        case ModelProvider.ANTHROPIC:
            return get_default_anthorpic_models_for_agents()
        case ModelProvider.OPENAI:
            return get_default_openai_models_for_agents()
        case ModelProvider.GEMINI:
            return get_default_gemini_models_for_agents()


def calculate_usage_cost(model: str, input_tokens: int, output_tokens: int):
    llm_model = get_model_by_name_or_snapshot(model)
    
    if not llm_model:
        return None
        
    input_cost = (input_tokens / 1_000_000) * llm_model.pricing.input_price
    output_cost = (output_tokens / 1_000_000) * llm_model.pricing.output_price
    return input_cost + output_cost