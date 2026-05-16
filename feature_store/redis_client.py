"""Async Redis helpers for SeqRec feature storage."""

from __future__ import annotations

from typing import Any, Protocol

from feature_store.schemas import (
    ItemFeatures,
    UserFeatures,
    item_features_from_hash,
    item_features_to_hash,
    user_features_from_hash,
    user_features_to_hash,
)


DEFAULT_USER_TTL_SECONDS = 24 * 60 * 60


class AsyncRedisHashClient(Protocol):
    """Subset of async Redis hash commands used by the feature store."""

    async def hset(self, name: str, mapping: dict[str, str]) -> Any:
        """Set a Redis hash mapping."""

    async def hgetall(self, name: str) -> dict[Any, Any]:
        """Return all fields for a Redis hash."""

    async def expire(self, name: str, time: int) -> Any:
        """Set key expiration in seconds."""


class RedisFeatureStore:
    """Typed feature-store access over an async Redis client."""

    def __init__(self, client: AsyncRedisHashClient, *, user_ttl_seconds: int | None = DEFAULT_USER_TTL_SECONDS) -> None:
        self.client = client
        self.user_ttl_seconds = user_ttl_seconds

    async def set_user_features(self, features: UserFeatures) -> None:
        key = user_key(features.user_id)
        await self.client.hset(key, mapping=user_features_to_hash(features))
        if self.user_ttl_seconds is not None:
            await self.client.expire(key, self.user_ttl_seconds)

    async def get_user_features(self, user_id: str) -> UserFeatures | None:
        payload = await self.client.hgetall(user_key(user_id))
        if not payload:
            return None
        return user_features_from_hash(payload)

    async def set_item_features(self, features: ItemFeatures) -> None:
        await self.client.hset(item_key(features.item_id), mapping=item_features_to_hash(features))

    async def get_item_features(self, item_id: str) -> ItemFeatures | None:
        payload = await self.client.hgetall(item_key(item_id))
        if not payload:
            return None
        return item_features_from_hash(payload)


def create_redis_feature_store(
    redis_url: str = "redis://localhost:6379/0",
    *,
    user_ttl_seconds: int | None = DEFAULT_USER_TTL_SECONDS,
) -> RedisFeatureStore:
    """Create a Redis-backed feature store using redis.asyncio when installed."""

    try:
        from redis import asyncio as redis_asyncio
    except ImportError as exc:
        raise ImportError("redis package is required for a real RedisFeatureStore client") from exc

    return RedisFeatureStore(
        redis_asyncio.from_url(redis_url, decode_responses=True),
        user_ttl_seconds=user_ttl_seconds,
    )


def user_key(user_id: str) -> str:
    return f"user:{user_id}"


def item_key(item_id: str) -> str:
    return f"item:{item_id}"
