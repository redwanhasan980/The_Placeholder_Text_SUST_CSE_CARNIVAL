from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .enums import CaseType, TransactionStatus, TransactionType
from .normalizer import TextFacts, normalize_phone
from .schemas import AnalyzeTicketRequest, Transaction


TYPE_BY_CASE: dict[CaseType, set[TransactionType]] = {
    CaseType.wrong_transfer: {TransactionType.transfer},
    CaseType.payment_failed: {TransactionType.payment},
    CaseType.refund_request: {TransactionType.payment, TransactionType.refund},
    CaseType.duplicate_payment: {TransactionType.payment},
    CaseType.merchant_settlement_delay: {TransactionType.settlement},
    CaseType.agent_cash_in_issue: {TransactionType.cash_in},
}


@dataclass
class MatchResult:
    transaction: Optional[Transaction]
    ambiguous: bool
    reason_codes: list[str]
    duplicate_pair: list[Transaction]


def find_relevant_transaction(
    request: AnalyzeTicketRequest,
    facts: TextFacts,
    case_type: CaseType,
) -> MatchResult:
    transactions = request.transaction_history or []
    if not transactions:
        return MatchResult(None, False, ["no_transaction_history"], [])

    direct = _match_by_transaction_id(transactions, facts)
    if direct:
        return MatchResult(direct, False, ["explicit_transaction_id"], [])

    if case_type == CaseType.duplicate_payment:
        duplicate_pair = find_duplicate_pair(transactions, facts)
        if duplicate_pair:
            return MatchResult(duplicate_pair[-1], False, ["duplicate_pattern"], duplicate_pair)

    if case_type == CaseType.phishing_or_social_engineering or case_type == CaseType.other:
        return MatchResult(None, False, ["no_specific_transaction"], [])

    scored = [(_score_transaction(tx, facts, case_type), tx) for tx in transactions]
    scored.sort(key=lambda item: item[0], reverse=True)
    top_score, top_tx = scored[0]
    if top_score < 8:
        return MatchResult(None, False, ["no_transaction_match"], [])

    close_matches = [(score, tx) for score, tx in scored if score >= 8 and top_score - score <= 2]
    if len(close_matches) > 1:
        has_strong_disambiguator = bool(facts.phones or facts.agent_ids or facts.merchant_ids or facts.mentioned_hour is not None)
        if not has_strong_disambiguator:
            return MatchResult(None, True, ["ambiguous_match"], [])

    return MatchResult(top_tx, False, ["transaction_match"], [])


def find_duplicate_pair(transactions: list[Transaction], facts: TextFacts) -> list[Transaction]:
    candidates = [
        tx
        for tx in transactions
        if tx.type == TransactionType.payment
        and tx.status == TransactionStatus.completed
        and (not facts.amounts or tx.amount in facts.amounts)
    ]
    candidates.sort(key=_timestamp_sort_key)
    for i, left in enumerate(candidates):
        for right in candidates[i + 1 :]:
            if left.amount == right.amount and left.counterparty == right.counterparty:
                return [left, right]
    return []


def count_prior_same_counterparty(transactions: list[Transaction], selected: Transaction) -> int:
    selected_time = _parse_timestamp(selected.timestamp)
    count = 0
    for tx in transactions:
        if tx.transaction_id == selected.transaction_id:
            continue
        if tx.type != selected.type or tx.counterparty != selected.counterparty:
            continue
        tx_time = _parse_timestamp(tx.timestamp)
        if selected_time is None or tx_time is None or tx_time <= selected_time:
            count += 1
    return count


def _match_by_transaction_id(transactions: list[Transaction], facts: TextFacts) -> Transaction | None:
    wanted = {tx_id.upper() for tx_id in facts.transaction_ids}
    for tx in transactions:
        if tx.transaction_id.upper() in wanted:
            return tx
    return None


def _score_transaction(tx: Transaction, facts: TextFacts, case_type: CaseType) -> int:
    score = 0
    if tx.amount in facts.amounts:
        score += 8
    if case_type in TYPE_BY_CASE and tx.type in TYPE_BY_CASE[case_type]:
        score += 5
    if _counterparty_matches(tx, facts):
        score += 8
    if _status_supports_case(tx.status, case_type):
        score += 4
    elif _status_contradicts_case(tx.status, case_type):
        score -= 2
    if _hour_matches(tx, facts):
        score += 3
    if not facts.amounts and case_type in TYPE_BY_CASE and tx.type in TYPE_BY_CASE[case_type]:
        score += 2
    return score


def _counterparty_matches(tx: Transaction, facts: TextFacts) -> bool:
    normalized_counterparty = normalize_phone(tx.counterparty)
    if normalized_counterparty in facts.phones:
        return True
    counterparty = tx.counterparty.upper()
    return counterparty in facts.agent_ids or counterparty in facts.merchant_ids


def _status_supports_case(status: TransactionStatus, case_type: CaseType) -> bool:
    if case_type == CaseType.wrong_transfer:
        return status == TransactionStatus.completed
    if case_type == CaseType.payment_failed:
        return status in {TransactionStatus.failed, TransactionStatus.pending}
    if case_type == CaseType.refund_request:
        return status in {TransactionStatus.completed, TransactionStatus.reversed}
    if case_type == CaseType.merchant_settlement_delay:
        return status == TransactionStatus.pending
    if case_type == CaseType.agent_cash_in_issue:
        return status in {TransactionStatus.pending, TransactionStatus.completed, TransactionStatus.failed}
    return status == TransactionStatus.completed


def _status_contradicts_case(status: TransactionStatus, case_type: CaseType) -> bool:
    if case_type == CaseType.payment_failed:
        return status == TransactionStatus.completed
    if case_type == CaseType.merchant_settlement_delay:
        return status == TransactionStatus.completed
    return False


def _hour_matches(tx: Transaction, facts: TextFacts) -> bool:
    if facts.mentioned_hour is None:
        return False
    tx_time = _parse_timestamp(tx.timestamp)
    if tx_time is None:
        return False
    return abs(tx_time.hour - facts.mentioned_hour) <= 1


def _timestamp_sort_key(tx: Transaction) -> datetime:
    parsed = _parse_timestamp(tx.timestamp)
    return parsed or datetime.min


def _parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

