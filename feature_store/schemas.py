"""Pydantic schemas and encoding helpers for Redis feature storage."""

from __future__ import annotations

from base64 import b64decode, b64encode
import json
import struct
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, Field, field_validator


USER_EMBEDDING_DIM = 128
ITEM_EMBEDDING_DIM = 128
BGE_EMBEDDING_DIM = 1024


class UserFeatures(BaseModel):
    """Online user features stored in Redis."""

    user_id: str
    interaction_history: list[int] = Field(default_factory=list)
    interaction_count: int
    top_categories: list[str] = Field(default_factory=list)
    last_seen_ts: int
    cold_start_flag: bool
    user_embedding: list[float] | None = None

    @field_validator("interaction_count", "last_seen_ts")
    @classmethod
    def _non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("value must be non-negative")
        return value

    @field_validator("user_embedding")
    @classmethod
    def _validate_user_embedding(cls, value: list[float] | None) -> list[float] | None:
        return _validate_embedding_dim(value, USER_EMBEDDING_DIM, "user_embedding")


class ItemFeatures(BaseModel):
    """Online item features stored in Redis."""

    item_id: str
    title: str
    category: str
    description: str = ""
    popularity_score: float
    item_embedding: list[float] | None = None
    bge_embedding: list[float] | None = None

    @field_validator("popularity_score")
    @classmethod
    def _non_negative_score(cls, value: float) -> float:
        if value < 0:
            raise ValueError("popularity_score must be non-negative")
        return value

    @field_validator("item_embedding")
    @classmethod
    def _validate_item_embedding(cls, value: list[float] | None) -> list[float] | None:
        return _validate_embedding_dim(value, ITEM_EMBEDDING_DIM, "item_embedding")

    @field_validator("bge_embedding")
    @classmethod
    def _validate_bge_embedding(cls, value: list[float] | None) -> list[float] | None:
        return _validate_embedding_dim(value, BGE_EMBEDDING_DIM, "bge_embedding")


def encode_float32_embedding(values: Sequence[float]) -> str:
    """Encode a float sequence as base64 little-endian float32 bytes."""

    floats = [float(value) for value in values]
    payload = struct.pack(f"<{len(floats)}f", *floats)
    return b64encode(payload).decode("ascii")


def decode_float32_embedding(payload: str) -> list[float]:
    """Decode a base64 little-endian float32 embedding."""

    raw = b64decode(payload.encode("ascii"))
    if len(raw) % 4 != 0:
        raise ValueError("embedding payload length must be divisible by 4")
    if not raw:
        return []
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def user_features_to_hash(features: UserFeatures) -> dict[str, str]:
    """Serialize user features to a Redis hash mapping."""

    payload = features.model_dump()
    return {
        "user_id": str(payload["user_id"]),
        "interaction_history": json.dumps(payload["interaction_history"]),
        "interaction_count": str(payload["interaction_count"]),
        "top_categories": json.dumps(payload["top_categories"]),
        "last_seen_ts": str(payload["last_seen_ts"]),
        "cold_start_flag": "1" if payload["cold_start_flag"] else "0",
        "user_embedding": "" if payload["user_embedding"] is None else encode_float32_embedding(payload["user_embedding"]),
    }


def user_features_from_hash(payload: Mapping[str | bytes, Any]) -> UserFeatures:
    """Deserialize user features from a Redis hash mapping."""

    normalized = _decode_hash(payload)
    embedding_payload = normalized.get("user_embedding", "")
    return UserFeatures(
        user_id=normalized["user_id"],
        interaction_history=json.loads(normalized["interaction_history"]),
        interaction_count=int(normalized["interaction_count"]),
        top_categories=json.loads(normalized["top_categories"]),
        last_seen_ts=int(normalized["last_seen_ts"]),
        cold_start_flag=normalized["cold_start_flag"] == "1",
        user_embedding=None if embedding_payload == "" else decode_float32_embedding(embedding_payload),
    )


def item_features_to_hash(features: ItemFeatures) -> dict[str, str]:
    """Serialize item features to a Redis hash mapping."""

    payload = features.model_dump()
    return {
        "item_id": str(payload["item_id"]),
        "title": str(payload["title"]),
        "category": str(payload["category"]),
        "description": str(payload["description"]),
        "popularity_score": str(payload["popularity_score"]),
        "item_embedding": "" if payload["item_embedding"] is None else encode_float32_embedding(payload["item_embedding"]),
        "bge_embedding": "" if payload["bge_embedding"] is None else encode_float32_embedding(payload["bge_embedding"]),
    }


def item_features_from_hash(payload: Mapping[str | bytes, Any]) -> ItemFeatures:
    """Deserialize item features from a Redis hash mapping."""

    normalized = _decode_hash(payload)
    item_embedding_payload = normalized.get("item_embedding", "")
    bge_embedding_payload = normalized.get("bge_embedding", "")
    return ItemFeatures(
        item_id=normalized["item_id"],
        title=normalized["title"],
        category=normalized["category"],
        description=normalized.get("description", ""),
        popularity_score=float(normalized["popularity_score"]),
        item_embedding=None if item_embedding_payload == "" else decode_float32_embedding(item_embedding_payload),
        bge_embedding=None if bge_embedding_payload == "" else decode_float32_embedding(bge_embedding_payload),
    )


def _decode_hash(payload: Mapping[str | bytes, Any]) -> dict[str, str]:
    decoded = {}
    for key, value in payload.items():
        decoded_key = key.decode("utf-8") if isinstance(key, bytes) else str(key)
        if isinstance(value, bytes):
            decoded[decoded_key] = value.decode("utf-8")
        else:
            decoded[decoded_key] = str(value)
    return decoded


def _validate_embedding_dim(values: list[float] | None, expected_dim: int, field_name: str) -> list[float] | None:
    if values is not None and len(values) != expected_dim:
        raise ValueError(f"{field_name} must have dimension {expected_dim}")
    return values
