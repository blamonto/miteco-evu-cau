from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Copernicus Data Space (CLC+ Backbone) ──
    copernicus_client_id: str = ""
    copernicus_client_secret: str = ""

    # ── EEA (HRL TCD — typically open access) ──
    eea_api_token: str = ""

    # ── Eurostat LAU ──
    lau_base_url: str = "https://gisco-services.ec.europa.eu/distribution/v2/lau/download/"
    lau_year: int = 2021

    # ── Cache ──
    cache_dir: str = "/tmp/miteco-cache"

    # ── Server ──
    host: str = "0.0.0.0"
    port: int = 8001
    log_level: str = "info"

    # ── Pixel resolution ──
    pixel_size_m: int = 10

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
