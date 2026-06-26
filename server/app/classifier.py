from .enums import CaseType
from .normalizer import TextFacts, has_any
from .schemas import AnalyzeTicketRequest


PHISHING_KEYWORDS = [
    "otp",
    "pin",
    "password",
    "passcode",
    "card number",
    "verification code",
    "security code",
    "scam",
    "fraud",
    "fake call",
    "phishing",
    "account will be blocked",
    "blocked if",
    "ওটিপি",
    "পিন",
    "পাসওয়ার্ড",
]

WRONG_TRANSFER_KEYWORDS = [
    "wrong number",
    "wrong person",
    "wrong recipient",
    "wrong account",
    "vul number",
    "vul nambar",
    "vul nambare",
    "bhul number",
    "bhul nambar",
    "bhul nambare",
    "vaul number",
    "vaul nambar",
    "typed it wrong",
    "by mistake",
    "mistake",
    "reverse it",
    "reverse",
    "sent to",
    "pathaisi",
    "pathaise",
    "pathalam",
    "pathailam",
    "pataisi",
    "didn't get it",
    "did not get it",
    "not received",
    "ভুল নাম্বার",
    "ভুল নম্বর",
    "ভুলে",
]

PAYMENT_FAILED_KEYWORDS = [
    "failed",
    "fail",
    "balance was deducted",
    "deducted",
    "kete geche",
    "recharge",
    "payment failed",
    "app showed failed",
    "কেটে",
    "ফেইল",
    "ব্যর্থ",
]

REFUND_KEYWORDS = [
    "refund",
    "money back",
    "return my money",
    "changed my mind",
    "don't want it",
    "dont want it",
    "ফেরত",
    "রিফান্ড",
]

DUPLICATE_KEYWORDS = [
    "twice",
    "two times",
    "double",
    "duplicate",
    "deducted twice",
    "charged twice",
    "duibar",
    "দুইবার",
    "২ বার",
]

MERCHANT_SETTLEMENT_KEYWORDS = [
    "merchant",
    "settlement",
    "settled",
    "sales",
    "batch",
    "next day",
    "মার্চেন্ট",
    "সেটেল",
]

AGENT_CASH_IN_KEYWORDS = [
    "cash in",
    "cash-in",
    "cashin",
    "agent",
    "balance",
    "not reflected",
    "did not receive",
    "didn't receive",
    "টাকা আসেনি",
    "ক্যাশ ইন",
    "এজেন্ট",
    "ব্যালেন্স",
]


def classify_case(request: AnalyzeTicketRequest, facts: TextFacts) -> tuple[CaseType, list[str], float]:
    text = facts.normalized
    reasons: list[str] = []

    if has_any(text, PHISHING_KEYWORDS) and has_any(text, ["asked", "share", "send", "called", "sms", "blocked", "চেয়েছে", "বলছে"]):
        return CaseType.phishing_or_social_engineering, ["phishing", "credential_risk"], 0.95

    if _looks_like_duplicate(request, facts):
        return CaseType.duplicate_payment, ["duplicate_payment"], 0.9

    if has_any(text, PAYMENT_FAILED_KEYWORDS) and has_any(text, ["payment", "pay", "paid", "recharge", "bill", "deducted", "balance", "পেমেন্ট"]):
        return CaseType.payment_failed, ["payment_failed"], 0.88

    if has_any(text, WRONG_TRANSFER_KEYWORDS) and has_any(text, ["sent", "transfer", "person", "number", "nambar", "nambare", "recipient", "brother", "reverse", "patha", "pata", "পাঠ", "নাম্বার", "নম্বর"]):
        return CaseType.wrong_transfer, ["wrong_transfer"], 0.86

    if has_any(text, REFUND_KEYWORDS):
        return CaseType.refund_request, ["refund_request"], 0.82

    if request.user_type == "merchant" or request.channel == "merchant_portal" or has_any(text, MERCHANT_SETTLEMENT_KEYWORDS):
        if has_any(text, ["settlement", "settled", "sales", "batch", "next day", "সেটেল"]):
            return CaseType.merchant_settlement_delay, ["merchant_settlement"], 0.9

    has_agent_or_cash_in = has_any(text, ["cash in", "cash-in", "cashin", "agent", "ক্যাশ ইন", "এজেন্ট"])
    has_non_receipt = has_any(text, ["not reflected", "not receive", "did not receive", "didn't receive", "balance", "টাকা আসেনি", "ব্যালেন্স"])
    if has_agent_or_cash_in and has_non_receipt:
        return CaseType.agent_cash_in_issue, ["agent_cash_in"], 0.86

    if has_any(text, WRONG_TRANSFER_KEYWORDS):
        return CaseType.wrong_transfer, ["possible_wrong_transfer"], 0.7

    return CaseType.other, ["vague_or_other"], 0.6


def _looks_like_duplicate(request: AnalyzeTicketRequest, facts: TextFacts) -> bool:
    text = facts.normalized
    if has_any(text, DUPLICATE_KEYWORDS):
        return True
    transactions = request.transaction_history
    for i, left in enumerate(transactions):
        for right in transactions[i + 1 :]:
            if (
                left.type == right.type
                and left.amount == right.amount
                and left.counterparty == right.counterparty
                and left.status == right.status
            ):
                if has_any(text, ["paid", "payment", "bill", "deducted", "charged"]):
                    return True
    return False
