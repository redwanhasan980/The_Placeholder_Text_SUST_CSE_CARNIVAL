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
        "request_received",
        {
            "ticket_id": request.ticket_id,
            "language": request.language,
            "channel": request.channel,
            "user_type": request.user_type,
            "transaction_count": len(request.transaction_history or []),
            "complaint": request.complaint,
            "llm_enabled": settings.use_llm,
            "confidence_threshold": settings.confidence_threshold,
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
    logger.step("rule_engine_decision", rule_snapshot)

    should_call_llm = settings.use_llm and confidence < settings.confidence_threshold
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
        if settings.debug_log_llm_prompt:
            logger.step("llm_prompt_messages", {"messages": llm_result.messages or []})
        if llm_result.error:
            logger.step("llm_error", {"error": llm_result.error})
        if settings.debug_log_llm_output:
            logger.step("llm_raw_output", {"raw_output": llm_result.raw_output})
        if llm_result.decision:
            logger.step("llm_parsed_decision", {"decision": llm_result.decision})
            (
                case_type,
                match,
                verdict,
                severity,
                department,
                human_review,
                classification_reasons,
                confidence,
                llm_texts,
            ) = _reconcile_llm_decision(
                request=request,
                facts=facts,
                current_case_type=case_type,
                current_match=match,
                current_verdict=verdict,
                current_severity=severity,
                current_department=department,
                current_human_review=human_review,
                classification_reasons=classification_reasons,
                current_confidence=confidence,
                llm_decision=llm_result.decision,
                logger=logger,
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
    logger.step("final_response", {"response": response})
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


def _reconcile_llm_decision(
    request: AnalyzeTicketRequest,
    facts,
    current_case_type: CaseType,
    current_match: MatchResult,
    current_verdict: EvidenceVerdict,
    current_severity: Severity,
    current_department: Department,
    current_human_review: bool,
    classification_reasons: list[str],
    current_confidence: float,
    llm_decision: LLMDecision,
    logger: DecisionLogger,
) -> tuple[CaseType, MatchResult, EvidenceVerdict, Severity, Department, bool, list[str], float, dict[str, str]]:
    accepted: dict[str, object] = {}
    rejected: dict[str, object] = {}

    if llm_decision.confidence < settings.llm_min_accept_confidence:
        logger.step(
            "llm_reconciliation",
            {
                "accepted": accepted,
                "rejected": {
                    "all": f"LLM confidence {llm_decision.confidence} below minimum {settings.llm_min_accept_confidence}"
                },
            },
        )
        return (
            current_case_type,
            current_match,
            current_verdict,
            current_severity,
            current_department,
            current_human_review,
            classification_reasons + ["llm_low_confidence_ignored"],
            current_confidence,
            {},
        )

    case_type = current_case_type
    if llm_decision.case_type and llm_decision.case_type != current_case_type:
        case_type = llm_decision.case_type
        accepted["case_type"] = {"from": current_case_type.value, "to": case_type.value}

    match = find_relevant_transaction(request, facts, case_type)
    if accepted.get("case_type"):
        accepted["rematched_for_llm_case_type"] = {
            "relevant_transaction_id": match.transaction.transaction_id if match.transaction else None,
            "ambiguous": match.ambiguous,
            "reason_codes": match.reason_codes,
        }

    llm_tx = _find_transaction(request, llm_decision.relevant_transaction_id)
    if llm_decision.relevant_transaction_id and llm_tx is None:
        rejected["relevant_transaction_id"] = "LLM returned an ID not present in transaction_history"
    elif llm_tx and _can_accept_llm_transaction(current_match, llm_decision):
        duplicate_pair = find_duplicate_pair(request.transaction_history, facts) if case_type == CaseType.duplicate_payment else []
        match = MatchResult(llm_tx, False, _dedupe(match.reason_codes + ["llm_transaction_assist"]), duplicate_pair)
        accepted["relevant_transaction_id"] = llm_tx.transaction_id

    verdict = decide_evidence_verdict(request, facts, case_type, match)
    if _can_accept_llm_verdict(case_type, match, verdict, llm_decision):
        verdict = llm_decision.evidence_verdict
        accepted["evidence_verdict"] = verdict.value
    elif llm_decision.evidence_verdict and llm_decision.evidence_verdict != verdict:
        rejected["evidence_verdict"] = {
            "llm": llm_decision.evidence_verdict.value,
            "deterministic": verdict.value,
            "reason": "deterministic evidence rules kept authority",
        }

    severity = decide_severity(case_type, verdict, match.transaction)
    if llm_decision.severity and _severity_rank(llm_decision.severity) > _severity_rank(severity):
        severity = llm_decision.severity
        accepted["severity"] = severity.value

    department = DEPARTMENT_BY_CASE[case_type]
    if llm_decision.department and llm_decision.department != department:
        rejected["department"] = {
            "llm": llm_decision.department.value,
            "deterministic": department.value,
            "reason": "department follows final case_type mapping",
        }

    human_review = decide_human_review(case_type, verdict, severity, match)
    if llm_decision.human_review_required:
        human_review = True
        accepted["human_review_required"] = True

    confidence = round(max(current_confidence, min(0.96, (current_confidence + llm_decision.confidence) / 2 + 0.08)), 2)
    reasons = _dedupe(
        classification_reasons
        + ["llm_assisted"]
        + [code.strip().lower().replace(" ", "_") for code in llm_decision.reason_codes if code.strip()]
    )
    texts = _accepted_llm_texts(llm_decision)
    if texts:
        accepted["text_fields"] = sorted(texts.keys())

    logger.step(
        "llm_reconciliation",
        {
            "accepted": accepted,
            "rejected": rejected,
            "final_case_type": case_type,
            "final_relevant_transaction_id": match.transaction.transaction_id if match.transaction else None,
            "final_evidence_verdict": verdict,
            "final_severity": severity,
            "final_department": department,
            "final_human_review_required": human_review,
            "final_confidence": confidence,
        },
    )

    return case_type, match, verdict, severity, department, human_review, reasons, confidence, texts


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


def _find_transaction(request: AnalyzeTicketRequest, transaction_id: str | None) -> Transaction | None:
    if not transaction_id:
        return None
    for tx in request.transaction_history:
        if tx.transaction_id == transaction_id:
            return tx
    return None


def _can_accept_llm_transaction(current_match: MatchResult, llm_decision: LLMDecision) -> bool:
    if llm_decision.confidence < 0.7:
        return False
    return current_match.transaction is None or current_match.ambiguous or "no_transaction_match" in current_match.reason_codes


def _can_accept_llm_verdict(
    case_type: CaseType,
    match: MatchResult,
    deterministic_verdict: EvidenceVerdict,
    llm_decision: LLMDecision,
) -> bool:
    if not llm_decision.evidence_verdict or llm_decision.evidence_verdict == deterministic_verdict:
        return False
    if llm_decision.confidence < 0.78:
        return False
    if case_type not in {CaseType.phishing_or_social_engineering, CaseType.other} and match.transaction is None:
        return False
    if match.ambiguous and llm_decision.evidence_verdict != EvidenceVerdict.insufficient_data:
        return False
    if deterministic_verdict == EvidenceVerdict.inconsistent and llm_decision.evidence_verdict == EvidenceVerdict.consistent:
        return False
    return deterministic_verdict == EvidenceVerdict.insufficient_data


def _accepted_llm_texts(llm_decision: LLMDecision) -> dict[str, str]:
    texts: dict[str, str] = {}
    if llm_decision.agent_summary and len(llm_decision.agent_summary.strip()) >= 20:
        texts["agent_summary"] = llm_decision.agent_summary.strip()
    if llm_decision.recommended_next_action and len(llm_decision.recommended_next_action.strip()) >= 20:
        texts["recommended_next_action"] = llm_decision.recommended_next_action.strip()
    if llm_decision.customer_reply and len(llm_decision.customer_reply.strip()) >= 20:
        texts["customer_reply"] = llm_decision.customer_reply.strip()
    return texts


def _severity_rank(severity: Severity) -> int:
    order = {
        Severity.low: 1,
        Severity.medium: 2,
        Severity.high: 3,
        Severity.critical: 4,
    }
    return order[severity]


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
