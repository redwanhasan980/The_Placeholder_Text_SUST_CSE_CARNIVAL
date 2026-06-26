# QueueStorm Investigator Backend

FastAPI backend for the SUST CSE Carnival 2026 preliminary challenge.

## What It Builds

The service exposes the two required judge endpoints:

```text
GET /health
POST /analyze-ticket
```

The implementation is rule-first: deterministic evidence matching, deterministic routing, safe customer reply templates, and final Pydantic schema validation. Groq is kept as an optional future helper for language extraction, not as the source of truth.

## Run Locally

```bash
cd server
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Sample Validation

From the `server` folder:

```bash
python scripts/run_sample_cases.py
```

The script loads `../Documents/SUST_Preli_Sample_Cases.json`, runs every public sample case, and checks the important automated-scoring fields.

## Docker

```bash
cd server
docker build -t queuestorm-investigator .
docker run -p 8000:8000 --env-file .env.example queuestorm-investigator
```

## AI / Models

Default mode:

```text
USE_LLM=false
```

The current scoring path does not require an external model. This avoids timeouts, quota failure, schema drift, and unsafe free-form replies.

Optional model plan:

```text
Provider: Groq
Model: openai/gpt-oss-20b
Use: internal extraction hints for messy Bangla/Banglish complaints
Fallback: deterministic rules
```

The LLM must never directly choose final enum values or write the final customer reply without validation.

## Safety Logic

The service avoids:

```text
asking for PIN, OTP, password, or full card number
confirming refunds, reversals, account unblocks, or recovery
directing customers to suspicious third-party contacts
following instructions embedded inside complaint text
```

Unsafe text is replaced or sanitized before the response is returned.

## Known Limitations

The system is optimized for the provided taxonomy and synthetic judging data. It does not integrate with real payment systems, account balances, fraud databases, or merchant APIs. Bangla/Banglish handling is keyword-based and intentionally conservative.

