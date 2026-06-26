import re
from dataclasses import dataclass, field
from typing import List


BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


@dataclass
class TextFacts:
    original: str
    normalized: str
    amounts: List[float] = field(default_factory=list)
    transaction_ids: List[str] = field(default_factory=list)
    phones: List[str] = field(default_factory=list)
    merchant_ids: List[str] = field(default_factory=list)
    agent_ids: List[str] = field(default_factory=list)
    mentioned_hour: int | None = None
    mentions_today: bool = False
    mentions_yesterday: bool = False


def normalize_phone(value: str) -> str:
    value = value.strip().replace(" ", "").replace("-", "")
    if value.startswith("+880"):
        return value
    if value.startswith("880"):
        return f"+{value}"
    if value.startswith("01"):
        return f"+88{value}"
    return value


def normalize_text(text: str) -> str:
    text = text.translate(BANGLA_DIGITS)
    text = text.lower()
    text = re.sub(r"[“”‘’]", "'", text)
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_facts(text: str) -> TextFacts:
    normalized = normalize_text(text)
    facts = TextFacts(original=text, normalized=normalized)
    facts.transaction_ids = [match.upper() for match in re.findall(r"\btxn[-_a-z0-9]*\d+\b", normalized, re.I)]
    facts.phones = sorted({normalize_phone(phone) for phone in re.findall(r"(?:\+?880|0)?1[3-9]\d{8}", normalized)})
    facts.merchant_ids = sorted({m.upper() for m in re.findall(r"\bmerchant[-_a-z0-9]*\b", normalized, re.I)})
    facts.agent_ids = sorted({a.upper() for a in re.findall(r"\bagent[-_a-z0-9]*\b", normalized, re.I)})
    facts.amounts = _extract_amounts(normalized)
    facts.mentioned_hour = _extract_hour(normalized)
    facts.mentions_today = any(token in normalized for token in ["today", "আজ", "aj"])
    facts.mentions_yesterday = any(token in normalized for token in ["yesterday", "গতকাল", "kalke", "kalker"])
    return facts


def _extract_amounts(text: str) -> List[float]:
    amounts: List[float] = []
    for match in re.finditer(r"(?<![a-z0-9+])\d+(?:\.\d+)?(?![a-z0-9])", text):
        value = match.group(0)
        start = match.start()
        before = text[max(0, start - 8) : start]
        after = text[match.end() : match.end() + 12]
        if "txn" in before:
            continue
        if len(value.split(".")[0]) >= 8:
            continue
        number = float(value)
        context = f"{before}{after}"
        amount_words = ["taka", "tk", "bdt", "৳", "টাকা", "amount", "sales", "bill", "paid", "sent", "pay"]
        if any(word in context for word in amount_words) or number >= 100:
            amounts.append(int(number) if number.is_integer() else number)
    return sorted(set(amounts))


def _extract_hour(text: str) -> int | None:
    match = re.search(r"\b([1-9]|1[0-2])\s*(am|pm)\b", text)
    if not match:
        return None
    hour = int(match.group(1))
    suffix = match.group(2)
    if suffix == "pm" and hour != 12:
        hour += 12
    if suffix == "am" and hour == 12:
        hour = 0
    return hour


def has_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)

