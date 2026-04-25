"""
Memory Graph — persists user travel preferences, past bookings, and aversions
in Redis (hot path) + SQLite/Postgres (cold path).
"""
import json
import redis
from sqlalchemy import create_engine, Column, String, JSON, DateTime
from sqlalchemy.orm import DeclarativeBase, Session
from datetime import datetime, timezone
from config.settings import get_settings

settings = get_settings()

# ─── Redis client ────────────────────────────────────────────────────────────
_redis = redis.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    password=settings.redis_password or None,
    db=settings.redis_db,
    decode_responses=True,
)

# ─── SQLAlchemy (cold storage) ───────────────────────────────────────────────
_engine = create_engine(settings.database_url, echo=False)


class Base(DeclarativeBase):
    pass


class UserMemoryRecord(Base):
    __tablename__ = "user_memory"
    user_id = Column(String, primary_key=True)
    profile = Column(JSON, default=dict)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


Base.metadata.create_all(_engine)


# ─── Public API ──────────────────────────────────────────────────────────────

def _redis_key(user_id: str) -> str:
    return f"memory:user:{user_id}"


def load_memory(user_id: str) -> dict:
    """Return user memory. Redis first; fall back to DB."""
    cached = _redis.get(_redis_key(user_id))
    if cached:
        return json.loads(cached)

    with Session(_engine) as session:
        record = session.get(UserMemoryRecord, user_id)
        if record:
            profile = record.profile or {}
            _redis.setex(_redis_key(user_id), settings.redis_session_ttl, json.dumps(profile))
            return profile
    return {}


def update_memory(user_id: str, updates: dict) -> None:
    """Merge updates into user memory and persist to both stores."""
    profile = load_memory(user_id)
    _deep_merge(profile, updates)

    _redis.setex(_redis_key(user_id), settings.redis_session_ttl, json.dumps(profile))

    with Session(_engine) as session:
        record = session.get(UserMemoryRecord, user_id)
        if record:
            record.profile = profile
            record.updated_at = datetime.now(timezone.utc)
        else:
            session.add(UserMemoryRecord(user_id=user_id, profile=profile))
        session.commit()


def record_bad_experience(user_id: str, destination: str, reason: str) -> None:
    update_memory(user_id, {"avoid_destinations": {destination: reason}})


def get_avoided_destinations(user_id: str) -> dict:
    return load_memory(user_id).get("avoid_destinations", {})


def _deep_merge(base: dict, updates: dict) -> None:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
