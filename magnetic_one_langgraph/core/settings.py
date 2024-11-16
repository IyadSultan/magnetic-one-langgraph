# config/settings.py

from typing import Dict, Any
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings and configuration."""

    # Required API Keys
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key")
    TAVILY_API_KEY: str = Field(..., description="Tavily API key")

    # System Settings
    DEBUG: bool = Field(False, description="Debug mode")
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    MAX_CONCURRENT_TASKS: int = Field(5, description="Maximum concurrent tasks")
    ENABLE_METRICS: bool = Field(True, description="Enable metrics collection")

    # LLM Settings
    LLM_MODEL: str = Field("gpt-4", description="LLM model name")
    LLM_TEMPERATURE: float = Field(0.7, description="LLM temperature")
    LLM_MAX_TOKENS: int = Field(2000, description="Maximum tokens for LLM")

    # Agent Settings
    MAX_ROUNDS: int = Field(20, description="Maximum interaction rounds")
    MAX_TIME: int = Field(3600, description="Maximum execution time in seconds")
    MAX_FILE_SIZE: int = Field(1048576, description="Maximum file size in bytes")
    MAX_MEMORY: int = Field(536870912, description="Maximum memory usage in bytes")

    # Project Paths
    PROJECT_ROOT: Path = Field(default=Path(__file__).parent.parent.resolve())
    LOG_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")

    # Derived Settings
    @property
    def LLM_SETTINGS(self) -> Dict[str, Any]:
        return {
            "model": self.LLM_MODEL,
            "temperature": self.LLM_TEMPERATURE,
            "max_tokens": self.LLM_MAX_TOKENS,
        }

    @property
    def AGENT_SETTINGS(self) -> Dict[str, Dict[str, Any]]:
        return {
            "orchestrator": {
                "max_rounds": self.MAX_ROUNDS,
                "max_time": self.MAX_TIME,
                "handle_messages_concurrently": False,
                "max_retries": 3,
            },
            "web_surfer": {
                "max_results": 5,
                "search_depth": "advanced",
                "timeout": 30,
                "max_retries": 3,
                "allowed_domains": [],
                "blocked_domains": [],
            },
            "file_surfer": {
                "max_file_size": self.MAX_FILE_SIZE,
                "supported_formats": [".txt", ".py", ".json", ".md", ".yaml", ".yml"],
                "max_files": 10,
                "timeout": 30,
            },
            "coder": {
                "max_code_length": 1000,
                "supported_languages": ["python", "javascript", "shell"],
                "timeout": 30,
                "max_complexity": 10,
                "style_check": True,
            },
            "terminal": {
                "timeout": 30,
                "max_memory": self.MAX_MEMORY,
                "allowed_modules": ["numpy", "pandas", "sklearn"],
                "blocked_modules": ["os", "subprocess", "sys"],
                "max_processes": 1,
            },
        }

    @property
    def SYSTEM_SETTINGS(self) -> Dict[str, Any]:
        return {
            "debug": self.DEBUG,
            "log_level": self.LOG_LEVEL,
            "max_concurrent_tasks": self.MAX_CONCURRENT_TASKS,
            "task_timeout": self.MAX_TIME,
            "enable_metrics": self.ENABLE_METRICS,
            "metrics_interval": 60,
        }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
