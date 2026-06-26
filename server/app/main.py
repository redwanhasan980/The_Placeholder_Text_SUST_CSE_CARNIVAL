from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .analyzer import analyze_ticket
from .config import settings
from .schemas import AnalyzeTicketRequest


app = FastAPI(title=settings.app_name)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze-ticket")
async def analyze_ticket_endpoint(payload: AnalyzeTicketRequest):
    return analyze_ticket(payload)


@app.exception_handler(ValidationError)
async def validation_exception_handler(_: Request, exc: ValidationError):
    return JSONResponse(status_code=422, content={"error": "Invalid request schema", "details": exc.errors()})


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "Internal analysis error"})

