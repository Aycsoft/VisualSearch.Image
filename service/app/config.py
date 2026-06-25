from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VISUAL_SEARCH_", case_sensitive=False)

    api_key: str = ""
    model_id: str = "facebook/dinov2-base"
    model_revision: str = "main"
    model_version: str = "dinov2-base-v1"
    device: str = "auto"
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None
    store_backend: str = "local"
    local_store_path: str = "./data/catalog"
    sqlite_path: str = "./data/visual-search.db"
    collection_prefix: str = "product_images"
    max_image_bytes: int = 128 * 1024 * 1024
    max_image_pixels: int = 80_000_000
    max_batch_size: int = Field(default=16, ge=1, le=64)
    max_batch_bytes: int = 256 * 1024 * 1024
    candidate_multiplier: int = Field(default=8, ge=2, le=50)
    high_confidence_score: float = Field(default=0.82, ge=0, le=1)
    medium_confidence_score: float = Field(default=0.70, ge=0, le=1)
    torch_threads: int = Field(default=4, ge=1, le=64)
    catalog_root: str = "/catalog"
    catalog_roots: str = ""
    allow_remove_missing: bool = False
    public_image_base_url: str = "/v1/catalog/image?path="

    @property
    def collection_name(self) -> str:
        safe_version = "".join(c if c.isalnum() or c == "_" else "_" for c in self.model_version)
        return f"{self.collection_prefix}_{safe_version}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
