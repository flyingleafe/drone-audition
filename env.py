from pydantic import BaseSettings


class Settings(BaseSettings):
    WORKSPACE_DIR: str
    DREGON_PATH: str
    SAMPLE_RATE: int


settings = Settings(_env_file=".env")
