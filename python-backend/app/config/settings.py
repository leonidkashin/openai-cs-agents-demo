from functools import lru_cache
from typing import List, Union, Optional
from urllib.parse import urljoin

from pydantic import (
    field_validator,
    MongoDsn,
    RedisDsn,
    AnyHttpUrl,
    BaseModel)
from pydantic_settings import BaseSettings, SettingsConfigDict

LOGGER_NAME = "naja"


class SearchServiceSettings(BaseModel):
    base_url: str
    api_key: str


class FlomniSettings(BaseModel):
    base_url: str
    api_key: str
    ai_tag_id: str
    operator_postback: str
    stats_token: Optional[str] = None


class EmailSettings(BaseModel):
    smtp_server: str = "smtp.yandex.ru"
    smtp_port: int = 587
    address: str = ""
    username: str = ""
    password: str = ""
    email: str = ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="ignore")

    APP_NAME: str = "mock_name"
    APP_ROOT_PATH: str = ""
    APP_DOCS_ENABLED: bool = False
    API_KEY: str

    SENTRY_DNS: Optional[str] = None
    ENVIRONMENT: Optional[str] = None

    SERVER_HOST: AnyHttpUrl = "http://localhost:8000"
    CORS_ORIGINS: List[AnyHttpUrl] = []

    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str = "https://915d7585-5257-4170-a934-5f19cf731965.qoo.qa"

    MONGODB_DSN: MongoDsn = 'mongodb://user:pass@localhost:27017/foobar'
    REDIS_DSN: RedisDsn = 'redis://user:pass@localhost:6379/1'
    REDIS_KEY_PREFIX: str

    USER_AGENT: str = 'NajaAI/1.0'
    HTTP_PROXY: Optional[str] = None

    NAJA_AI_HISTORY_LIMIT: int = 1000
    NAJA_AI_HISTORY_DAYS: int = 30

    PROMPT_SERVICE_TYPE: str = 'local'  # local, db

    ACCEPT_EVERY_N: int = 1
    flomni: Optional[FlomniSettings] = None

    DAYS_TO_START_INACTIVE_CHAT: Optional[int] = 3

    SERVICE_WORK_MODE: Optional[str] = "suggest"  # self, suggest
    BOT_IMAGE: Optional[str] = "https://naja.ams3.digitaloceanspaces.com/restore/profile.jpg"

    search_service: SearchServiceSettings
    email: EmailSettings = EmailSettings()

    # noinspection PyNestedDecorators
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    @property
    def base_path(self):
        return urljoin(str(self.SERVER_HOST), self.APP_ROOT_PATH)


@lru_cache()
def get_settings() -> Settings:
    return Settings()
