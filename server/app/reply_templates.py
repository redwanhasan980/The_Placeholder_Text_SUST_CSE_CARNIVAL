from .enums import CaseType, EvidenceVerdict
from .schemas import AnalyzeTicketRequest, Transaction


def build_texts(
    request: AnalyzeTicketRequest,
    case_type: CaseType,
    verdict: EvidenceVerdict,
    transaction: Transaction | None,
    reason_codes: list[str],
) -> tuple[str, str, str]:
    is_bn = request.language == "bn"
    if is_bn:
        return _build_bn_texts(request, case_type, verdict, transaction)
    return _build_en_texts(request, case_type, verdict, transaction, reason_codes)


def _tx_label(transaction: Transaction | None) -> str:
    return f"transaction {transaction.transaction_id}" if transaction else "the reported issue"


def _build_en_texts(
    request: AnalyzeTicketRequest,
    case_type: CaseType,
    verdict: EvidenceVerdict,
    transaction: Transaction | None,
    reason_codes: list[str],
) -> tuple[str, str, str]:
    tx_label = _tx_label(transaction)
    amount = f"{transaction.amount:g} BDT" if transaction else "the reported amount"

    if case_type == CaseType.phishing_or_social_engineering:
        summary = "Customer reports a suspicious contact or message involving sensitive credentials such as OTP, PIN, or password."
        action = "Escalate to fraud_risk, log the reported pattern, and remind the customer that official support never asks for secret credentials."
        reply = "Thank you for reaching out. We never ask for your PIN, OTP, or password under any circumstances. Please do not share these with anyone, even if they claim to be from us. Our fraud team has been notified of this incident."
    elif case_type == CaseType.wrong_transfer:
        if transaction:
            summary = f"Customer reports a possible wrong transfer involving {amount} via {transaction.transaction_id} to {transaction.counterparty}."
            if verdict == EvidenceVerdict.inconsistent:
                summary += " Prior activity with the same counterparty suggests the claim needs careful verification."
            action = f"Verify {transaction.transaction_id} with the customer and handle through the wrong-transfer dispute workflow."
            reply = f"We have received your request regarding transaction {transaction.transaction_id}. Please do not share your PIN or OTP with anyone. Our dispute team will review the case and contact you through official support channels."
        else:
            summary = "Customer reports a possible wrong transfer, but the provided history does not identify one clear matching transaction."
            action = "Ask for the recipient number or transaction ID before initiating any dispute workflow."
            reply = "Thank you for reaching out. We need the recipient number or transaction ID to identify the correct transfer. Please do not share your PIN or OTP with anyone."
    elif case_type == CaseType.payment_failed:
        summary = f"Customer reports a failed payment or recharge with possible balance deduction for {tx_label}."
        action = f"Investigate ledger status for {tx_label} and start the standard reversal flow only if the payment is eligible."
        reply = f"We have noted that {_tx_label(transaction)} may have caused an unexpected balance deduction. Our payments team will review the case and any eligible amount will be returned through official channels. Please do not share your PIN or OTP with anyone."
    elif case_type == CaseType.refund_request:
        summary = f"Customer requests a refund for {tx_label}. The request depends on transaction status and merchant or service policy."
        action = "Check refund eligibility and guide the customer through the applicable official process without promising a refund."
        if transaction and str(transaction.counterparty).upper().startswith("MERCHANT"):
            reply = "Thank you for reaching out. Refunds for completed merchant payments depend on the merchant's own policy. We recommend contacting the merchant directly. If you need help reaching them, please reply and we will guide you. Please do not share your PIN or OTP with anyone."
        else:
            reply = f"We have received your refund request for {_tx_label(transaction)}. Our team will review eligibility and update you through official support channels. Please do not share your PIN or OTP with anyone."
    elif case_type == CaseType.duplicate_payment:
        summary = f"Customer reports a possible duplicate payment. {tx_label.capitalize()} appears to be the likely duplicate based on the provided history."
        action = f"Verify the duplicate with payments_ops and the biller or merchant before any eligible reversal is processed."
        reply = f"We have noted the possible duplicate payment for {_tx_label(transaction)}. Our payments team will verify it and any eligible amount will be returned through official channels. Please do not share your PIN or OTP with anyone."
    elif case_type == CaseType.merchant_settlement_delay:
        summary = f"Merchant reports a delayed settlement for {tx_label}."
        action = "Route to merchant_operations to check settlement batch status and communicate the expected settlement time."
        reply = f"We have noted your concern about {_tx_label(transaction)}. Our merchant operations team will check the batch status and update you through official channels."
    elif case_type == CaseType.agent_cash_in_issue:
        summary = f"Customer reports a cash-in issue through an agent for {tx_label}; balance may not be reflected."
        action = f"Route to agent_operations to verify the cash-in state for {_tx_label(transaction)} and resolve according to the standard cash-in SLA."
        reply = f"We have noted your concern about {_tx_label(transaction)}. Our agent operations team will review it and contact you through official support channels. Please do not share your PIN or OTP with anyone."
    else:
        summary = "Customer reports a vague or uncategorized money concern without enough detail to identify a specific transaction."
        action = "Ask for the transaction ID, amount, approximate time, and a short description of what went wrong."
        reply = "Thank you for reaching out. To help you faster, please share the transaction ID, the amount involved, and a short description of what went wrong. Please do not share your PIN or OTP with anyone."

    if "ambiguous_match" in reason_codes:
        summary = "Customer complaint could refer to multiple similar transactions, so the correct transaction cannot be determined safely."
        action = "Ask for a disambiguating detail such as recipient number, merchant, or transaction ID before taking action."
        reply = "Thank you for reaching out. We see multiple similar transactions. Please share the recipient number, merchant, or transaction ID so we can identify the right one. Please do not share your PIN or OTP with anyone."

    return summary, action, reply


def _build_bn_texts(
    request: AnalyzeTicketRequest,
    case_type: CaseType,
    verdict: EvidenceVerdict,
    transaction: Transaction | None,
) -> tuple[str, str, str]:
    tx_id = transaction.transaction_id if transaction else "উল্লেখিত লেনদেন"
    if case_type == CaseType.agent_cash_in_issue:
        summary = f"গ্রাহক {tx_id} সম্পর্কিত এজেন্ট ক্যাশ-ইন সমস্যা জানিয়েছেন; ব্যালেন্সে টাকা প্রতিফলিত হয়নি বলে অভিযোগ।"
        action = f"{tx_id} এর অবস্থা এজেন্ট অপারেশনস টিম দিয়ে যাচাই করান এবং প্রযোজ্য SLA অনুযায়ী সমাধান করুন।"
        reply = f"আপনার লেনদেন {tx_id} এর বিষয়ে আমরা অবগত হয়েছি। আমাদের এজেন্ট অপারেশনস দল এটি যাচাই করবে এবং অফিসিয়াল চ্যানেলে আপনাকে জানাবে। অনুগ্রহ করে কারও সাথে আপনার পিন বা ওটিপি শেয়ার করবেন না।"
    elif case_type == CaseType.phishing_or_social_engineering:
        summary = "গ্রাহক ওটিপি, পিন বা পাসওয়ার্ড চাওয়া সন্দেহজনক কল বা মেসেজের কথা জানিয়েছেন।"
        action = "তাৎক্ষণিকভাবে fraud_risk টিমে এস্কেলেট করুন এবং গ্রাহককে গোপন তথ্য শেয়ার না করার পরামর্শ দিন।"
        reply = "আমাদের সাথে যোগাযোগ করার জন্য ধন্যবাদ। আমরা কখনও আপনার পিন, ওটিপি বা পাসওয়ার্ড চাই না। কেউ আমাদের পরিচয়ে চাইলেও এগুলো শেয়ার করবেন না। আমাদের ফ্রড টিম বিষয়টি পর্যালোচনা করবে।"
    else:
        summary = f"গ্রাহক {tx_id} সম্পর্কিত অভিযোগ জানিয়েছেন।"
        action = "লেনদেনের তথ্য যাচাই করে অফিসিয়াল চ্যানেলে গ্রাহককে আপডেট দিন।"
        reply = f"আপনার অভিযোগ আমরা পেয়েছি। {tx_id} সম্পর্কিত তথ্য আমাদের দল যাচাই করবে এবং অফিসিয়াল চ্যানেলে আপনাকে জানাবে। অনুগ্রহ করে কারও সাথে আপনার পিন বা ওটিপি শেয়ার করবেন না।"
    return summary, action, reply

