from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg://aqi:aqi@localhost:5432/air_quality"
    openaq_api_key: str = ""
    openaq_base_url: str = "https://api.openaq.org/v3"
    open_meteo_base_url: str = "https://api.open-meteo.com/v1/forecast"


settings = Settings()
