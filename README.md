# PHI De-Identification Pipeline

A hybrid (rule-first) pipeline that detects and redacts **Protected Health
Information** from free-text clinical notes, built on
[Microsoft Presidio](https://github.com/microsoft/presidio).

**Why this matters:** a generative model trained on inadequately de-identified
notes can reproduce patient names and details at inference time. De-identification
is what makes real clinical data *safe to train on* — it's the gate in front of
every healthcare-AI product, not a compliance afterthought. This repo targets
HIPAA **Safe Harbor** (the 18 identifier categories) and measures the
precision/recall tradeoff that decides whether de-identified data stays useful.

This is the **Phase 1 baseline**: a high-precision rule layer plus an evaluation
harness, scaffolded so the Phase 2 ML layer drops straight in.

## Quickstart

This project uses [uv](https://docs.astral.sh/uv/). Install it, then:

```bash
make install          # uv sync — create .venv and install from uv.lock
make model            # uv run python -m spacy download en_core_web_sm  (enables the ML layer)
make test             # uv run pytest
make eval             # benchmark on synthetic labeled notes
make api              # uv run uvicorn phi_deid.api:app --reload
```

Redact a file or stdin:

```bash
echo "Patient Jane Doe (MRN: 44820193) seen 03/14/2025." | uv run phi-deid deidentify
```

## How it works

```
raw note → [preprocess] → [detect: rules + ML] → [reconcile] → [redact] → de-identified note
                                                                    └→ [evaluate] (offline)
```

- **Detection** is hybrid. High-precision regex recognizers handle structured
  identifiers (`recognizers.py`: MRN, account, beneficiary/member, device IDs,
  dates); Presidio's defaults add SSN, email, phone, IP, etc.; and spaCy NER
  (once the model is installed) catches free-text names and locations.
- **Redaction** (Phase 1) replaces each span with a `<ENTITY_TYPE>` tag. The
  seam for Phase 3 utility-preserving operators (surrogates, date-shifting)
  lives in `pipeline.deidentify`.
- **Evaluation** (`evaluate.py`) reports token-level leak rate, per-identifier
  recall, and entity-level overlap metrics.

### Rules-only vs. full mode

Without the spaCy model the pipeline runs in **rules-only mode** (a blank
tokenizer drives the rule layer). This is intentional: it makes the baseline
runnable anywhere, and the eval makes the gap visible.

```
Token-level   precision 76.5%   recall 56.9%   LEAK RATE 43.1%
Per-type recall: SSN/email/MRN/account/beneficiary/dates = 100%,
                 phone 89%,  PERSON 0%,  LOCATION 0%
```

The rules nail every structured identifier and miss free-text names/locations
entirely — which is precisely the motivation for the **Phase 2** ML layer.
Install the model (`make model`) and PERSON/LOCATION recall climbs, dropping the
leak rate. Quantifying that lift over this baseline is the Phase 2 deliverable.

> Note: in rules-only mode a phone number may be mislabeled (e.g. `UK_NHS`) by a
> built-in recognizer. It's still redacted, so it's not a leak — type accuracy
> is a separate concern from the safety metric.

## Project layout

```
src/phi_deid/
  nlp.py           spaCy engine + blank fallback
  recognizers.py   custom HIPAA recognizers
  pipeline.py      detect / deidentify
  synthetic.py     labeled synthetic note generator
  evaluate.py      metrics + report
  cli.py           phi-deid deidentify | eval
  api.py           FastAPI service
scripts/run_eval.py
tests/
data/              i2b2/n2c2 corpus goes here (gitignored)
```

## Data

The synthetic generator (`synthetic.py`) produces labeled notes with exact
ground-truth spans, so the benchmark runs immediately. For the real benchmark,
request the **i2b2/n2c2 2014 de-identification** corpus (DUA required) and drop
it in `data/` — which is gitignored, because real PHI never belongs in version
control.

## Roadmap

- **Phase 1 (this repo):** rule layer + eval harness + service/CLI. ✅
- **Phase 2:** clinical NER (scispaCy / a transformer) as a Presidio recognizer;
  span reconciliation; quantify recall lift over the baseline.
- **Phase 3:** utility-preserving redaction (consistent surrogates, date-shifting);
  audit log + low-confidence review queue.
- **Phase 4:** React/TypeScript demo UI on the FastAPI endpoint.
- **Phase 5:** memorization experiment — show a model trained on unredacted vs.
  redacted notes leaks PHI at inference.

## Limitations

A regex-first baseline over-redacts some labels and misses free-text PHI without
the ML layer. De-identification is never perfect; the honest goal is high recall
(low leak rate) with measured, improving utility.
