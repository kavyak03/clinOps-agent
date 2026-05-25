# Production Hardening Notes

This repository is still a research-grade biomedical AI system, but it includes several production-minded safeguards.

## Implemented hardening

### 1. Provider error handling
OpenAI and Anthropic provider wrappers catch provider failures such as quota, billing, model-access, timeout, or malformed JSON errors and return schema-compatible fallback responses instead of crashing the API.

### 2. Strict LLM output schema
All provider responses are normalized through `src/llm/schema.py`.

The canonical schema is:

```json
{
  "answer": "string",
  "citations": [{"id": "1", "doc_id": "source"}],
  "uncertainties": ["string"],
  "refusal_reason": null,
  "provider_error": null
}
```

This prevents common LLM issues such as:
- `citations` returned as `["1", "2"]`
- `uncertainties` returned as a single string
- non-JSON provider output

### 3. CI eval gate
`src/eval/eval_gate.py` provides thresholded evaluation gating.

Run locally without Docker:

```bash
python -m src.eval.eval_gate --smoke
```

Run against a live API:

```bash
python -m src.eval.eval_gate --api-base http://localhost:8000 --endpoint /agent/ask
```

Thresholds live in:

```text
eval/quality_thresholds.json
```

### 4. Optional API key auth
Set:

```env
REQUIRE_API_KEY=true
API_KEY=your_internal_key
```

Then call endpoints with:

```bash
curl -H "X-API-Key: your_internal_key" ...
```

Local default remains:

```env
REQUIRE_API_KEY=false
```

### 5. Production mode disables synthetic fallback
In local mode, `/decision/ask` can generate a synthetic cohort if none is provided.

In production mode:

```env
APP_ENV=production
```

`/decision/ask` requires explicit cohort input and will return an error if `cohort` is omitted.

This prevents accidental decision outputs based on demo/synthetic data.

## Still not production-complete

Here are some suggested additions before real deployment:
- full authentication and authorization
- PHI redaction before logging
- database migrations
- idempotent ingestion / corpus versioning
- stronger validation metrics
- stronger evaluation datasets
- API rate limiting
- structured monitoring dashboards
