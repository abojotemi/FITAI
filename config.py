# config.py
from dataclasses import dataclass
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AppConfig:
    VALID_IMAGE_TYPES: List[str] = ("png", "jpg", "jpeg", "gif")
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    CACHE_TTL: int = 3600  # 1 hour
    DEFAULT_LANG: str = "en"
    
config = AppConfig()