from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # Anthropic
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY")
    claude_model: str = Field("claude-sonnet-4-6", env="CLAUDE_MODEL")

    # Amadeus (Flight + Hotel)
    amadeus_api_key: str = Field(..., env="AMADEUS_API_KEY")
    amadeus_api_secret: str = Field(..., env="AMADEUS_API_SECRET")
    amadeus_base_url: str = Field("https://test.api.amadeus.com", env="AMADEUS_BASE_URL")

    # Google Maps
    google_maps_api_key: str = Field(..., env="GOOGLE_MAPS_API_KEY")
    google_distance_matrix_url: str = Field(
        "https://maps.googleapis.com/maps/api/distancematrix/json",
        env="GOOGLE_DISTANCE_MATRIX_URL",
    )

    # Razorpay
    razorpay_key_id: str = Field(..., env="RAZORPAY_KEY_ID")
    razorpay_key_secret: str = Field(..., env="RAZORPAY_KEY_SECRET")
    razorpay_currency: str = Field("INR", env="RAZORPAY_CURRENCY")
    razorpay_webhook_secret: str = Field(..., env="RAZORPAY_WEBHOOK_SECRET")

    # Pinecone
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    pinecone_environment: str = Field("gcp-starter", env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field("travel-knowledge-base", env="PINECONE_INDEX_NAME")

    # Redis
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_password: str = Field("", env="REDIS_PASSWORD")
    redis_db: int = Field(0, env="REDIS_DB")
    redis_lock_timeout: int = Field(30, env="REDIS_LOCK_TIMEOUT")
    redis_session_ttl: int = Field(86400, env="REDIS_SESSION_TTL")

    # Database
    database_url: str = Field("sqlite:///./travel_system.db", env="DATABASE_URL")

    # MCP
    mcp_server_host: str = Field("0.0.0.0", env="MCP_SERVER_HOST")
    mcp_server_port: int = Field(8000, env="MCP_SERVER_PORT")
    mcp_secret_key: str = Field(..., env="MCP_SECRET_KEY")
    mcp_circuit_breaker_threshold: int = Field(5, env="MCP_CIRCUIT_BREAKER_THRESHOLD")
    mcp_circuit_breaker_timeout: int = Field(60, env="MCP_CIRCUIT_BREAKER_TIMEOUT")

    # Hotel search config
    hotel_search_radius: int = Field(5, env="HOTEL_SEARCH_RADIUS")
    hotel_max_results: int = Field(20, env="HOTEL_MAX_RESULTS")

    # Train (RailAPI via RapidAPI)
    railapi_key: str = Field("", env="RAILAPI_KEY")
    railapi_base_url: str = Field(
        "https://indian-railway1.p.rapidapi.com", env="RAILAPI_BASE_URL"
    )

    # Bus (RedBus via RapidAPI)
    redbus_api_key: str = Field("", env="REDBUS_API_KEY")
    redbus_base_url: str = Field("https://redbus.p.rapidapi.com", env="REDBUS_BASE_URL")

    # Weather
    openweather_api_key: str = Field("", env="OPENWEATHER_API_KEY")
    openweather_base_url: str = Field(
        "https://api.openweathermap.org/data/2.5", env="OPENWEATHER_BASE_URL"
    )

    # App
    app_env: str = Field("development", env="APP_ENV")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    max_itinerary_retries: int = Field(3, env="MAX_ITINERARY_RETRIES")
    human_in_loop_timeout: int = Field(300, env="HUMAN_IN_LOOP_TIMEOUT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
