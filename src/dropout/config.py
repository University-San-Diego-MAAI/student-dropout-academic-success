"""Application configuration via pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    artifact_dir: Path = Path("artifacts")
    model_path: Path = Path("artifacts/model.pkl")
    preprocessor_path: Path = Path("artifacts/preprocessor.pkl")
    label_encoder_path: Path = Path("artifacts/label_encoder.pkl")
    log_level: str = "INFO"


def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
