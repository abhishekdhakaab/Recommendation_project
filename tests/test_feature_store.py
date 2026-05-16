import asyncio

import pytest

from data.feature_store_init import initialize_feature_store
from feature_store.redis_client import RedisFeatureStore, item_key, user_key
from feature_store.schemas import (
    BGE_EMBEDDING_DIM,
    ITEM_EMBEDDING_DIM,
    USER_EMBEDDING_DIM,
    ItemFeatures,
    UserFeatures,
    decode_float32_embedding,
    encode_float32_embedding,
    item_features_from_hash,
    item_features_to_hash,
    user_features_from_hash,
    user_features_to_hash,
)


def test_embedding_base64_round_trip() -> None:
    values = [0.0, 1.5, -2.25]

    decoded = decode_float32_embedding(encode_float32_embedding(values))

    assert decoded == pytest.approx(values)


def test_user_features_hash_round_trip() -> None:
    features = UserFeatures(
        user_id="U1",
        interaction_history=[1, 2, 3],
        interaction_count=3,
        top_categories=["Beauty", "Sports"],
        last_seen_ts=123,
        cold_start_flag=True,
        user_embedding=[0.1] * USER_EMBEDDING_DIM,
    )

    payload = user_features_to_hash(features)
    restored = user_features_from_hash({key.encode(): value.encode() for key, value in payload.items()})

    assert payload["interaction_history"] == "[1, 2, 3]"
    assert payload["cold_start_flag"] == "1"
    assert restored.user_id == features.user_id
    assert restored.interaction_history == features.interaction_history
    assert restored.user_embedding == pytest.approx(features.user_embedding)


def test_item_features_hash_round_trip() -> None:
    features = ItemFeatures(
        item_id="I1",
        title="Item",
        category="Beauty",
        description="A useful item",
        popularity_score=0.7,
        item_embedding=[0.2] * ITEM_EMBEDDING_DIM,
        bge_embedding=[0.3] * BGE_EMBEDDING_DIM,
    )

    restored = item_features_from_hash(item_features_to_hash(features))

    assert restored.item_id == "I1"
    assert restored.title == "Item"
    assert restored.popularity_score == pytest.approx(0.7)
    assert restored.item_embedding == pytest.approx(features.item_embedding)
    assert restored.bge_embedding == pytest.approx(features.bge_embedding)


def test_redis_feature_store_get_set_helpers() -> None:
    asyncio.run(_assert_redis_feature_store_get_set_helpers())


async def _assert_redis_feature_store_get_set_helpers() -> None:
    client = FakeAsyncRedis()
    store = RedisFeatureStore(client, user_ttl_seconds=60)
    user = UserFeatures(
        user_id="U1",
        interaction_history=[7, 8],
        interaction_count=2,
        top_categories=["Home"],
        last_seen_ts=456,
        cold_start_flag=True,
        user_embedding=None,
    )
    item = ItemFeatures(
        item_id="I1",
        title="Item",
        category="Home",
        description="",
        popularity_score=1.2,
        item_embedding=None,
        bge_embedding=None,
    )

    await store.set_user_features(user)
    await store.set_item_features(item)

    assert await store.get_user_features("U1") == user
    assert await store.get_item_features("I1") == item
    assert client.expirations[user_key("U1")] == 60
    assert item_key("I1") in client.hashes
    assert await store.get_user_features("missing") is None


def test_initialize_feature_store_writes_users_and_items() -> None:
    asyncio.run(_assert_initialize_feature_store_writes_users_and_items())


async def _assert_initialize_feature_store_writes_users_and_items() -> None:
    store = RedisFeatureStore(FakeAsyncRedis(), user_ttl_seconds=None)

    stats = await initialize_feature_store(
        store,
        users=[
            {
                "user_id": "U1",
                "interaction_history": [],
                "interaction_count": 0,
                "top_categories": [],
                "last_seen_ts": 0,
                "cold_start_flag": True,
            }
        ],
        items=[
            {
                "item_id": "I1",
                "title": "Item",
                "category": "Beauty",
                "description": "",
                "popularity_score": 0.5,
            }
        ],
    )

    assert stats == {"users": 1, "items": 1}
    assert (await store.get_user_features("U1")).cold_start_flag is True
    assert (await store.get_item_features("I1")).category == "Beauty"


def test_schema_validates_embedding_dimensions() -> None:
    with pytest.raises(ValueError, match="user_embedding must have dimension"):
        UserFeatures(
            user_id="U1",
            interaction_history=[],
            interaction_count=0,
            top_categories=[],
            last_seen_ts=0,
            cold_start_flag=True,
            user_embedding=[0.1],
        )


class FakeAsyncRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.expirations: dict[str, int] = {}

    async def hset(self, name: str, mapping: dict[str, str]) -> None:
        self.hashes[name] = dict(mapping)

    async def hgetall(self, name: str) -> dict[str, str]:
        return dict(self.hashes.get(name, {}))

    async def expire(self, name: str, time: int) -> None:
        self.expirations[name] = time
