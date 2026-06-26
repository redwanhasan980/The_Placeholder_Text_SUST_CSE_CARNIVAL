# QueueStorm Investigator Backend

FastAPI backend for the SUST CSE Carnival 2026 preliminary challenge.

## What It Builds

The service exposes the two required judge endpoints:

```text
GET /health
POST /analyze-ticket
```

The implementation is hybrid: deterministic evidence matching, deterministic routing, safe customer reply templates, optional Groq LLM assistance for low-confidence cases, and final Pydantic schema validation.

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

The scoring path works without an external model. When `USE_LLM=true`, the backend calls Groq only if the rule engine confidence is below the configured threshold.

Optional model plan:

```text
Provider: Groq
Model: openai/gpt-oss-20b
Use: low-confidence decision assistance for messy English, Bangla, Banglish, spelling mistakes, and ambiguous complaints
Fallback: deterministic rules
```

The LLM receives the full ticket context, transaction history, preliminary rule decision, taxonomy, evidence rules, and fintech safety rules. It returns a strict JSON proposal. The backend may accept parts of that proposal, but final schema validation, department mapping, and safety sanitation remain deterministic.

Useful `.env` controls:

```text
USE_LLM=true
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-20b
confidence=0.75
LLM_MIN_ACCEPT_CONFIDENCE=0.55
LLM_TIMEOUT_SECONDS=4
```

If the rule confidence is below `confidence`, the LLM assist layer is called. If the LLM confidence is below `LLM_MIN_ACCEPT_CONFIDENCE`, its decision is ignored.

## Debug Logging

Detailed per-ticket logging can be enabled from `.env`:

```text
DEBUG_LOG_ENABLED=true
DEBUG_LOG_TO_CONSOLE=true
DEBUG_LOG_FILE=logs/decision_debug.log
DEBUG_LOG_LLM_PROMPT=true
DEBUG_LOG_LLM_OUTPUT=true
```

The log shows:

```text
request metadata and complaint
normalized extracted facts
rule-engine decision
LLM gate decision
full LLM prompt messages, if enabled
raw LLM output, if enabled
parsed LLM decision
accepted/rejected LLM fields
final response
```

Logs may include synthetic complaint text and LLM output. Do not enable verbose logs for real customer data.

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
