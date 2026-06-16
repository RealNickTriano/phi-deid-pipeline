"""Minimal FastAPI service wrapping the pipeline.

    uvicorn phi_deid.api:app --reload

POST /deidentify  {"text": "..."}  ->  {"redacted": "...", "entities": [...]}

This is the seam the React/TypeScript demo UI (Phase 4) talks to.
"""

from __future__ import annotations

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from .pipeline import deidentify

app = FastAPI(title="PHI De-Identification API", version="0.1.0")


class DeidRequest(BaseModel):
    text: str


class EntityOut(BaseModel):
    entity_type: str
    start: int
    end: int
    score: float


class DeidResponse(BaseModel):
    redacted: str
    entities: List[EntityOut]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/deidentify", response_model=DeidResponse)
def deidentify_endpoint(req: DeidRequest) -> DeidResponse:
    redacted, entities = deidentify(req.text)
    return DeidResponse(
        redacted=redacted,
        entities=[
            EntityOut(
                entity_type=e.entity_type, start=e.start, end=e.end, score=e.score
            )
            for e in entities
        ],
    )
