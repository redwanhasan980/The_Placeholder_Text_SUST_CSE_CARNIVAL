import re


SAFE_FALLBACK_REPLY_EN = (
    "We have noted your concern. Our team will review the matter through official support channels. "
    "Please do not share your PIN, OTP, password, or sensitive account information with anyone."
)

SAFE_FALLBACK_ACTION = "Review the case through official support channels and avoid requesting sensitive credentials."


UNSAFE_CREDENTIAL_REQUESTS = [
    r"\b(share|send|provide|give|tell|submit|enter)\s+(us|me|our team|support)?\s*(your\s+)?(pin|otp|password|passcode|full card number)\b",
    r"\b(pin|otp|password|passcode)\s+(please|required|needed|din|den|dao)\b",
]

UNSAFE_PROMISES = [
    r"\bwe will refund\b",
    r"\bwe will reverse\b",
    r"\brefund (is )?confirmed\b",
    r"\breversal (is )?confirmed\b",
    r"\bwe have reversed\b",
    r"\bwe have refunded\b",
    r"\baccount (is )?unblocked\b",
    r"\bmoney will be refunded\b",
]

SUSPICIOUS_THIRD_PARTY = [
    r"\bcontact\s+\+?\d{8,}\b",
    r"\bcall\s+\+?\d{8,}\b",
    r"\bwhatsapp\s+\+?\d{8,}\b",
]


def sanitize_response_text(customer_reply: str, recommended_next_action: str) -> tuple[str, str, list[str]]:
    reasons: list[str] = []
    if _has_unsafe_credential_request(customer_reply):
        customer_reply = SAFE_FALLBACK_REPLY_EN
        reasons.append("credential_request_sanitized")
    if _has_unsafe_promise(customer_reply) or _has_unsafe_promise(recommended_next_action):
        customer_reply = _replace_unsafe_promises(customer_reply)
        recommended_next_action = _replace_unsafe_promises(recommended_next_action)
        reasons.append("financial_promise_sanitized")
    if _matches_any(customer_reply, SUSPICIOUS_THIRD_PARTY):
        customer_reply = SAFE_FALLBACK_REPLY_EN
        reasons.append("third_party_contact_sanitized")
    return customer_reply, recommended_next_action, reasons


def _has_unsafe_credential_request(text: str) -> bool:
    lowered = text.lower()
    if "do not share" in lowered or "never share" in lowered:
        return False
    return _matches_any(lowered, UNSAFE_CREDENTIAL_REQUESTS)


def _has_unsafe_promise(text: str) -> bool:
    return _matches_any(text.lower(), UNSAFE_PROMISES)


def _replace_unsafe_promises(text: str) -> str:
    safe = "any eligible amount will be returned through official channels"
    replacements = [
        (r"\bwe will refund (you|your money|the amount)?\b", safe),
        (r"\bwe will reverse (it|the transaction)?\b", "the team will review whether the transaction is eligible for reversal"),
        (r"\bmoney will be refunded\b", safe),
        (r"\brefund (is )?confirmed\b", "refund eligibility will be reviewed"),
        (r"\breversal (is )?confirmed\b", "reversal eligibility will be reviewed"),
        (r"\bwe have reversed\b", "the team will review"),
        (r"\bwe have refunded\b", "the team will review"),
        (r"\baccount (is )?unblocked\b", "account status will be reviewed"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in patterns)

