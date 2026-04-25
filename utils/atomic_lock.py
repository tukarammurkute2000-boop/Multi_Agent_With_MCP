"""
Redis-based atomic locking to prevent double-booking.
Uses SET NX EX (atomic compare-and-set).
"""
import uuid
import time
import redis
from contextlib import contextmanager
from config.settings import get_settings

settings = get_settings()

_redis = redis.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    password=settings.redis_password or None,
    db=settings.redis_db,
    decode_responses=True,
)


class LockAcquisitionError(Exception):
    pass


@contextmanager
def booking_lock(resource_id: str, timeout: int = None):
    """
    Context manager that holds an atomic lock on resource_id.
    resource_id: e.g. "flight:GOI:2025-05-10:AI502:seat:12A"
    Raises LockAcquisitionError if lock cannot be acquired.
    """
    timeout = timeout or settings.redis_lock_timeout
    lock_key = f"lock:{resource_id}"
    token = str(uuid.uuid4())

    acquired = _redis.set(lock_key, token, nx=True, ex=timeout)
    if not acquired:
        raise LockAcquisitionError(f"Resource {resource_id!r} is already locked")

    try:
        yield
    finally:
        # Only release if we still own the lock (Lua script = atomic check+delete)
        lua = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        _redis.eval(lua, 1, lock_key, token)


def is_locked(resource_id: str) -> bool:
    return _redis.exists(f"lock:{resource_id}") == 1
