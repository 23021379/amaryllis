from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


ALL_API_KEYS = [
    "[REDACTED_BY_SCRIPT]",#1
    "[REDACTED_BY_SCRIPT]",#2
    "[REDACTED_BY_SCRIPT]",#3
    "[REDACTED_BY_SCRIPT]",#4
    "[REDACTED_BY_SCRIPT]",#5
    "[REDACTED_BY_SCRIPT]",#6
    "[REDACTED_BY_SCRIPT]",#7
    "[REDACTED_BY_SCRIPT]",#8
    "[REDACTED_BY_SCRIPT]",#9
    "[REDACTED_BY_SCRIPT]",#10
    "[REDACTED_BY_SCRIPT]",#11
    "[REDACTED_BY_SCRIPT]",#12
    "[REDACTED_BY_SCRIPT]",#13
    "[REDACTED_BY_SCRIPT]",#14
    "[REDACTED_BY_SCRIPT]",#15
    "[REDACTED_BY_SCRIPT]",#16
    "[REDACTED_BY_SCRIPT]",#17
    "[REDACTED_BY_SCRIPT]",#18
    "[REDACTED_BY_SCRIPT]",#19
    "[REDACTED_BY_SCRIPT]",#20
    "[REDACTED_BY_SCRIPT]",#21
    "[REDACTED_BY_SCRIPT]",#22
    "[REDACTED_BY_SCRIPT]",#23
    "[REDACTED_BY_SCRIPT]",#24
    "[REDACTED_BY_SCRIPT]"#25
]
class Settings(BaseSettings):
    """
    Centralized, environment-aware application configuration.
    """
    gemini_api_keys: list[str] = Field(default=ALL_API_KEYS)
    log_level: str = Field('INFO', alias='LOG_LEVEL')
    validation_timeout_ms: int = Field(30000, alias='[REDACTED_BY_SCRIPT]')
    validation_retries: int = Field(2, alias='VALIDATION_RETRIES') # Default to 2 retries (3 total attempts)
    max_correction_attempts: int = Field(3, alias='[REDACTED_BY_SCRIPT]') # Default to 3 correction attempts
    
    # Fallback chain: 2.5 Flash (Fastest/New) -> 1.5 Flash (Reliable) -> 1.5 Pro (Heavy Duty)
    gemini_models: list[str] = Field(
        default=["[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]", "gemini-2.5-flash", "[REDACTED_BY_SCRIPT]"],
        alias='GEMINI_MODELS'
    )

    @field_validator('gemini_api_keys', mode='before')
    @classmethod
    def _parse_comma_separated_str(cls, v: str) -> list[str]:
        if isinstance(v, str):
            keys = [key.strip() for key in v.split(',') if key.strip()]
            if not keys:
                raise ValueError("[REDACTED_BY_SCRIPT]")
            return keys
        return v

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = True

# This line acts as a fail-fast gate. If config is invalid, the app will
# crash on import, which is the desired behavior.
settings = Settings()