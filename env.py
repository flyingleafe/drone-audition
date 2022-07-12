from pydantic import BaseSettings
from dotenv import find_dotenv


class Settings(BaseSettings):
    WORKSPACE_DIR: str
    DREGON_PATH: str
    SAMPLE_RATE: int


settings = Settings(_env_file=find_dotenv(".env"))
