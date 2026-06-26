from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from .enums import (
    CaseType,
    Channel,
    Department,
    EvidenceVerdict,
    Language,
    Severity,
    TransactionStatus,
    TransactionType,
    UserType,
)


class Transaction(BaseModel):
    transaction_id: str
    timestamp: str
    type: TransactionType
    amount: float
    counterparty: str
    status: TransactionStatus


class AnalyzeTicketRequest(BaseModel):
    ticket_id: str
    complaint: str
    language: Optional[Language] = None
    channel: Optional[Channel] = None
    user_type: Optional[UserType] = None
    campaign_context: Optional[str] = None
    transaction_history: List[Transaction] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("ticket_id")
    def ticket_id_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("ticket_id must not be empty")
        return value.strip()

    @validator("complaint")
    def complaint_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("complaint must not be empty")
        return value.strip()


class AnalyzeTicketResponse(BaseModel):
    ticket_id: str
    relevant_transaction_id: Optional[str]
    evidence_verdict: EvidenceVerdict
    case_type: CaseType
    severity: Severity
    department: Department
    agent_summary: str
    recommended_next_action: str
    customer_reply: str
    human_review_required: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason_codes: List[str] = Field(default_factory=list)

