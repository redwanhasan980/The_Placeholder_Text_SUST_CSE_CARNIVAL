from .classifier import classify_case
from .config import settings
from .debug_logger import DecisionLogger
from .enums import CaseType, Department, EvidenceVerdict, Severity, TransactionStatus
from .llm_client import LLMDecision, request_llm_decision
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
    logger = DecisionLogger(request.ticket_id)
    logger.step(
        "request_data",
        {
            "request": _model_to_dict(request),
            "transaction_count": len(request.transaction_history or []),
            "runtime": {
                "llm_enabled": settings.use_llm,
                "confidence_threshold": settings.confidence_threshold,
                "llm_min_accept_confidence": settings.llm_min_accept_confidence,
                "groq_model": settings.groq_model,
            },
        },
    )

    facts = extract_facts(request.complaint)
    logger.step(
        "text_facts_extracted",
        {
            "normalized": facts.normalized,
            "amounts": facts.amounts,
            "transaction_ids": facts.transaction_ids,
            "phones": facts.phones,
            "merchant_ids": facts.merchant_ids,
            "agent_ids": facts.agent_ids,
            "mentioned_hour": facts.mentioned_hour,
            "mentions_today": facts.mentions_today,
            "mentions_yesterday": facts.mentions_yesterday,
        },
    )

    case_type, classification_reasons, classification_confidence = classify_case(request, facts)
    match = find_relevant_transaction(request, facts, case_type)
    verdict = decide_evidence_verdict(request, facts, case_type, match)
    severity = decide_severity(case_type, verdict, match.transaction)
    department = DEPARTMENT_BY_CASE[case_type]
    human_review = decide_human_review(case_type, verdict, severity, match)
    confidence = _confidence(classification_confidence, verdict, match)
    llm_texts: dict[str, str] = {}
    llm_status = {
        "sent_to_llm": False,
        "status": "skipped",
        "reason": "LLM gate has not been evaluated yet.",
        "parsed": False,
        "error": None,
        "plain_text_response": "",
    }

    rule_snapshot = _decision_snapshot(
        case_type=case_type,
        match=match,
        verdict=verdict,
        severity=severity,
        department=department,
        human_review=human_review,
        confidence=confidence,
        reason_codes=classification_reasons + match.reason_codes,
    )
    logger.step(
        "rule_based_decision",
        {
            "classification_confidence": classification_confidence,
            "classification_reasons": classification_reasons,
            "decision": rule_snapshot,
        },
    )

    should_call_llm = settings.use_llm and confidence < settings.confidence_threshold
    llm_status["sent_to_llm"] = should_call_llm
    if not should_call_llm:
        llm_status["reason"] = "Rule confidence is high enough or USE_LLM is disabled."
    logger.step(
        "llm_gate",
        {
            "should_call_llm": should_call_llm,
            "rule_confidence": confidence,
            "threshold": settings.confidence_threshold,
            "use_llm": settings.use_llm,
        },
    )

    if should_call_llm:
        llm_result = request_llm_decision(request, facts, rule_snapshot)
        llm_status.update(
            {
                "status": "success" if llm_result.decision else "failed",
                "reason": "LLM returned a valid parsed decision." if llm_result.decision else "LLM did not return a usable parsed decision.",
                "parsed": bool(llm_result.decision),
                "error": llm_result.error,
                "plain_text_response": llm_result.raw_output,
            }
        )
        logger.step(
            "llm_prompt",
            {
                "sent_to_llm": True,
                "prompt_logging_enabled": settings.debug_log_llm_prompt,
                "messages": llm_result.messages or [] if settings.debug_log_llm_prompt else None,
            },
        )
        logger.step(
            "llm_response",
            {
                "sent_to_llm": True,
                "status": llm_status["status"],
                "output_logging_enabled": settings.debug_log_llm_output,
                "plain_text_response": llm_result.raw_output if settings.debug_log_llm_output else None,
                "error": llm_result.error,
                "parsed_decision": llm_result.decision,
            },
        )
        if llm_result.decision:
            response = _response_from_llm_decision(request, llm_result.decision)
            logger.step(
                "final_decision_making",
                {
                    "decision_source": "llm_exact",
                    "message": "LLM returned valid parsed JSON. Backend is returning LLM fields exactly without rule-based rewrite, sanitizer, remap, or template override.",
                    "llm_status": llm_status,
                },
            )
            logger.step(
                "output",
                {
                    "response": response,
                    "llm_response_status": llm_status,
                },
            )
            logger.close()
            return response
    else:
        logger.step(
            "llm_prompt",
            {
                "sent_to_llm": False,
                "messages": None,
                "reason": "Rule confidence is high enough or USE_LLM is disabled.",
            },
        )
        logger.step(
            "llm_response",
            {
                "sent_to_llm": False,
                "status": llm_status["status"],
                "plain_text_response": None,
                "parsed_decision": None,
                "error": None,
            },
        )

    reason_codes = _dedupe(classification_reasons + match.reason_codes + _verdict_reason_codes(verdict, case_type, match))
    summary, next_action, customer_reply = build_texts(request, case_type, verdict, match.transaction, reason_codes)
    if llm_texts:
        summary = llm_texts.get("agent_summary") or summary
        next_action = llm_texts.get("recommended_next_action") or next_action
        customer_reply = llm_texts.get("customer_reply") or customer_reply
        reason_codes.append("llm_text_assist")

    customer_reply, next_action, safety_reasons = sanitize_response_text(customer_reply, next_action)
    reason_codes = _dedupe(reason_codes + safety_reasons)
    logger.step(
        "final_decision_making",
        {
            "final_case_type": case_type,
            "final_relevant_transaction_id": match.transaction.transaction_id if match.transaction else None,
            "final_evidence_verdict": verdict,
            "final_severity": severity,
            "final_department": department,
            "final_human_review_required": human_review,
            "final_confidence": confidence,
            "reason_codes": reason_codes,
            "safety_sanitizer_reason_codes": safety_reasons,
            "llm_text_fields_used": sorted(llm_texts.keys()),
            "llm_status": llm_status,
            "decision_priority": "rule_based_only_because_llm_was_skipped_or_failed",
        },
    )

    response = AnalyzeTicketResponse(
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
    logger.step(
        "output",
        {
            "response": response,
            "llm_response_status": llm_status,
        },
    )
    logger.close()
    return response


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


def _response_from_llm_decision(request: AnalyzeTicketRequest, llm_decision: LLMDecision) -> AnalyzeTicketResponse:
    return AnalyzeTicketResponse(
        ticket_id=request.ticket_id,
        relevant_transaction_id=llm_decision.relevant_transaction_id,
        evidence_verdict=llm_decision.evidence_verdict,
        case_type=llm_decision.case_type,
        severity=llm_decision.severity,
        department=llm_decision.department,
        agent_summary=llm_decision.agent_summary,
        recommended_next_action=llm_decision.recommended_next_action,
        customer_reply=llm_decision.customer_reply,
        human_review_required=llm_decision.human_review_required,
        confidence=llm_decision.confidence,
        reason_codes=llm_decision.reason_codes,
    )


def _decision_snapshot(
    case_type: CaseType,
    match: MatchResult,
    verdict: EvidenceVerdict,
    severity: Severity,
    department: Department,
    human_review: bool,
    confidence: float,
    reason_codes: list[str],
) -> dict[str, object]:
    return {
        "case_type": case_type.value,
        "relevant_transaction_id": match.transaction.transaction_id if match.transaction else None,
        "match_ambiguous": match.ambiguous,
        "match_reason_codes": match.reason_codes,
        "evidence_verdict": verdict.value,
        "severity": severity.value,
        "department": department.value,
        "human_review_required": human_review,
        "confidence": confidence,
        "reason_codes": reason_codes,
    }


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


def _model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
