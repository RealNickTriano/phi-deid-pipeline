"""Evaluation harness.

Three views, because de-identification has a safety asymmetry: a missed
identifier is a PHI *leak*, while over-redaction only costs data utility.

1. Token-level (headline): treat every token as PHI / not-PHI and score
   precision, recall, F1. Recall is the safety number; we report
   ``leak_rate = 1 - recall`` explicitly.
2. Per-gold-type recall: for each gold entity type, what fraction of spans were
   covered by *any* predicted span. This is the "rules nail SSN/email, miss
   names" story - and the motivation for the Phase 2 ML layer.
3. Entity-level overlap precision/recall (type-agnostic), as a coarse summary.

All matching is overlap-based on character offsets, so it is robust to the
boundary differences you get when a labeled pattern (e.g. "MRN: 12345678")
slightly overshoots the gold value span.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

_TOKEN_RE = re.compile(r"\S+")

# A span here is just (start, end, entity_type).
SpanT = Tuple[int, int, str]


def _char_mask(length: int, spans: Sequence[SpanT]) -> bytearray:
    mask = bytearray(length)
    for start, end, _ in spans:
        for i in range(max(0, start), min(length, end)):
            mask[i] = 1
    return mask


def _overlaps(a: SpanT, b: SpanT) -> bool:
    return a[0] < b[1] and b[0] < a[1]


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _token_counts(text: str, gold: Sequence[SpanT], pred: Sequence[SpanT]) -> Counts:
    gmask = _char_mask(len(text), gold)
    pmask = _char_mask(len(text), pred)
    c = Counts()
    for m in _TOKEN_RE.finditer(text):
        s, e = m.start(), m.end()
        gold_phi = any(gmask[i] for i in range(s, e))
        pred_phi = any(pmask[i] for i in range(s, e))
        if gold_phi and pred_phi:
            c.tp += 1
        elif pred_phi and not gold_phi:
            c.fp += 1
        elif gold_phi and not pred_phi:
            c.fn += 1
    return c


def _entity_counts(gold: Sequence[SpanT], pred: Sequence[SpanT]) -> Counts:
    c = Counts()
    c.tp = sum(1 for g in gold if any(_overlaps(g, p) for p in pred))
    c.fn = len(gold) - c.tp
    c.fp = sum(1 for p in pred if not any(_overlaps(p, g) for g in gold))
    return c


@dataclass
class Report:
    token: Counts
    entity: Counts
    per_type: Dict[str, Counts]
    n_notes: int

    @property
    def leak_rate(self) -> float:
        return 1.0 - self.token.recall


def evaluate(
    notes_spans: Sequence[Tuple[str, Sequence[SpanT], Sequence[SpanT]]],
) -> Report:
    """Aggregate metrics over (text, gold_spans, pred_spans) triples."""
    token = Counts()
    entity = Counts()
    per_type: Dict[str, Counts] = defaultdict(Counts)

    for text, gold, pred in notes_spans:
        token.add(_token_counts(text, gold, pred))
        entity.add(_entity_counts(gold, pred))
        # per-gold-type recall
        by_type: Dict[str, List[SpanT]] = defaultdict(list)
        for g in gold:
            by_type[g[2]].append(g)
        for etype, gspans in by_type.items():
            covered = sum(1 for g in gspans if any(_overlaps(g, p) for p in pred))
            per_type[etype].tp += covered
            per_type[etype].fn += len(gspans) - covered

    return Report(token=token, entity=entity, per_type=dict(per_type), n_notes=len(notes_spans))


def format_report(report: Report) -> str:
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append(f"PHI DE-IDENTIFICATION - PHASE 1 BASELINE  ({report.n_notes} notes)")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Token-level (PHI vs not) -- the headline safety metric")
    lines.append(f"  precision : {report.token.precision:6.1%}")
    lines.append(f"  recall    : {report.token.recall:6.1%}")
    lines.append(f"  f1        : {report.token.f1:6.1%}")
    lines.append(f"  LEAK RATE : {report.leak_rate:6.1%}   (= 1 - recall)")
    lines.append("")
    lines.append("Entity-level overlap (type-agnostic)")
    lines.append(f"  precision : {report.entity.precision:6.1%}")
    lines.append(f"  recall    : {report.entity.recall:6.1%}")
    lines.append(f"  f1        : {report.entity.f1:6.1%}")
    lines.append("")
    lines.append("Per-gold-type recall  (which identifiers the rules catch)")
    for etype in sorted(report.per_type):
        c = report.per_type[etype]
        total = c.tp + c.fn
        lines.append(f"  {etype:<24} {c.recall:6.1%}  ({c.tp}/{total})")
    lines.append("=" * 60)
    return "\n".join(lines)
