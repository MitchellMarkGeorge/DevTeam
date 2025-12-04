from typing import Literal
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from devteam.config import utils

WebSearchOptions = Literal["exa", "firecrawl"]
EnviromentOptions = Literal["dev", "prod"]
ModeOptions = Literal["local", "remote"]

DEFAULT_CONFIG_FILE_PATH = "~/.devteam/config.yaml"


def default_config_file(data) -> Path:
    return (
        Path(DEFAULT_CONFIG_FILE_PATH)
        if data["enviroment"] == "prod"
        else utils.get_relative_path("../../config.yaml")
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=utils.get_relative_path("../../.env"), env_ignore_empty=True
    )

    enviroment: EnviromentOptions = "dev"
    config_file: Path = Field(default_factory=default_config_file)
    mode: ModeOptions = "local"

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemeni_api_key: str | None = None

    firecrawl_api: str | None = None
    exa_api: str | None = None

    web_search: WebSearchOptions | None = None


settings = Settings()
