from .classifier import classify_case
from .enums import CaseType, Department, EvidenceVerdict, Severity, TransactionStatus
from .matcher import MatchResult, count_prior_same_counterparty, find_duplicate_pair, find_relevant_transaction
from .normalizer import extract_facts
from .reply_templates import build_texts
from .safety import sanitize_response_text
from .schemas import AnalyzeTicketRequest, AnalyzeTicketResponse, Transaction


DEPARTMENT_BY_CASE = {
    CaseType.wrong_transfer: Department.dispute_resolution,
    CaseType.payment_failed: Department.payments_ops,
    CaseType.refund_request: Department.customer_support,
    CaseType.duplicate_payment: Department.payments_ops,
    CaseType.merchant_settlement_delay: Department.merchant_operations,
    CaseType.agent_cash_in_issue: Department.agent_operations,
    CaseType.phishing_or_social_engineering: Department.fraud_risk,
    CaseType.other: Department.customer_support,
}


def analyze_ticket(request: AnalyzeTicketRequest) -> AnalyzeTicketResponse:
    facts = extract_facts(request.complaint)
    case_type, classification_reasons, classification_confidence = classify_case(request, facts)
    match = find_relevant_transaction(request, facts, case_type)
    verdict = decide_evidence_verdict(request, facts, case_type, match)
    severity = decide_severity(case_type, verdict, match.transaction)
    department = DEPARTMENT_BY_CASE[case_type]
    human_review = decide_human_review(case_type, verdict, severity, match)

    reason_codes = _dedupe(classification_reasons + match.reason_codes + _verdict_reason_codes(verdict, case_type, match))
    summary, next_action, customer_reply = build_texts(request, case_type, verdict, match.transaction, reason_codes)
    customer_reply, next_action, safety_reasons = sanitize_response_text(customer_reply, next_action)
    reason_codes = _dedupe(reason_codes + safety_reasons)

    confidence = _confidence(classification_confidence, verdict, match)

    return AnalyzeTicketResponse(
        ticket_id=request.ticket_id,
        relevant_transaction_id=match.transaction.transaction_id if match.transaction else None,
        evidence_verdict=verdict,
        case_type=case_type,
        severity=severity,
        department=department,
        agent_summary=summary,
        recommended_next_action=next_action,
        customer_reply=customer_reply,
        human_review_required=human_review,
        confidence=confidence,
        reason_codes=reason_codes,
    )


def decide_evidence_verdict(
    request: AnalyzeTicketRequest,
    facts,
    case_type: CaseType,
    match: MatchResult,
) -> EvidenceVerdict:
    tx = match.transaction
    if case_type == CaseType.phishing_or_social_engineering:
        return EvidenceVerdict.insufficient_data
    if match.ambiguous or tx is None:
        return EvidenceVerdict.insufficient_data
    if case_type == CaseType.wrong_transfer:
        if count_prior_same_counterparty(request.transaction_history, tx) >= 2:
            return EvidenceVerdict.inconsistent
        if tx.status == TransactionStatus.completed:
            return EvidenceVerdict.consistent
        return EvidenceVerdict.inconsistent
    if case_type == CaseType.payment_failed:
        if tx.status in {TransactionStatus.failed, TransactionStatus.pending}:
            return EvidenceVerdict.consistent
        return EvidenceVerdict.inconsistent
    if case_type == CaseType.refund_request:
        return EvidenceVerdict.consistent
    if case_type == CaseType.duplicate_payment:
        return EvidenceVerdict.consistent if match.duplicate_pair or find_duplicate_pair(request.transaction_history, facts) else EvidenceVerdict.inconsistent
    if case_type == CaseType.merchant_settlement_delay:
        return EvidenceVerdict.consistent if tx.status == TransactionStatus.pending else EvidenceVerdict.inconsistent
    if case_type == CaseType.agent_cash_in_issue:
        return EvidenceVerdict.consistent
    return EvidenceVerdict.insufficient_data


def decide_severity(case_type: CaseType, verdict: EvidenceVerdict, tx: Transaction | None) -> Severity:
    amount = tx.amount if tx else 0
    if case_type == CaseType.phishing_or_social_engineering:
        return Severity.critical
    if case_type == CaseType.wrong_transfer:
        return Severity.high if amount >= 5000 else Severity.medium
    if case_type == CaseType.payment_failed:
        return Severity.high
    if case_type == CaseType.duplicate_payment:
        return Severity.high
    if case_type == CaseType.agent_cash_in_issue:
        return Severity.high
    if case_type == CaseType.merchant_settlement_delay:
        return Severity.medium
    if case_type == CaseType.refund_request:
        return Severity.medium if amount >= 5000 else Severity.low
    if verdict == EvidenceVerdict.inconsistent:
        return Severity.medium
    return Severity.low


def decide_human_review(
    case_type: CaseType,
    verdict: EvidenceVerdict,
    severity: Severity,
    match: MatchResult,
) -> bool:
    if case_type == CaseType.phishing_or_social_engineering:
        return True
    if verdict == EvidenceVerdict.inconsistent:
        return True
    if case_type == CaseType.wrong_transfer and not match.ambiguous:
        return True
    if case_type == CaseType.duplicate_payment and match.transaction is not None:
        return True
    if case_type == CaseType.agent_cash_in_issue:
        return True
    return False


def _verdict_reason_codes(verdict: EvidenceVerdict, case_type: CaseType, match: MatchResult) -> list[str]:
    reasons = [verdict.value]
    if verdict == EvidenceVerdict.inconsistent:
        reasons.append("evidence_inconsistent")
    if verdict == EvidenceVerdict.insufficient_data:
        reasons.append("needs_clarification" if match.ambiguous or case_type != CaseType.phishing_or_social_engineering else "safety_only_case")
    return reasons


def _confidence(classification_confidence: float, verdict: EvidenceVerdict, match: MatchResult) -> float:
    confidence = classification_confidence
    if match.transaction:
        confidence += 0.04
    if match.ambiguous:
        confidence -= 0.16
    if verdict == EvidenceVerdict.insufficient_data:
        confidence -= 0.08
    if verdict == EvidenceVerdict.inconsistent:
        confidence -= 0.04
    return round(max(0.35, min(0.96, confidence)), 2)


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result

