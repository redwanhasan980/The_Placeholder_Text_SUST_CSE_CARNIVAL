import json
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from app.analyzer import analyze_ticket
from app.schemas import AnalyzeTicketRequest


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PATH = ROOT / "Documents" / "SUST_Preli_Sample_Cases.json"


IMPORTANT_FIELDS = [
    "relevant_transaction_id",
    "evidence_verdict",
    "case_type",
    "severity",
    "department",
    "human_review_required",
]


def main() -> int:
    data = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    failures = []
    for case in data["cases"]:
        request = AnalyzeTicketRequest(**case["input"])
        actual = _to_dict(analyze_ticket(request))
        expected = case["expected_output"]
        mismatches = {
            field: {"expected": expected[field], "actual": actual[field]}
            for field in IMPORTANT_FIELDS
            if actual[field] != expected[field]
        }
        if mismatches:
            failures.append((case["id"], mismatches))
        print(f"{case['id']}: {actual['case_type']} / {actual['evidence_verdict']} / {actual['relevant_transaction_id']}")

    if failures:
        print("\nMISMATCHES")
        for case_id, mismatches in failures:
            print(case_id, json.dumps(mismatches, ensure_ascii=False, indent=2))
        return 1

    output_dir = Path(__file__).resolve().parents[1] / "sample_outputs"
    output_dir.mkdir(exist_ok=True)
    first_case = data["cases"][0]
    first_output = _to_dict(analyze_ticket(AnalyzeTicketRequest(**first_case["input"])))
    (output_dir / "sample_01_output.json").write_text(json.dumps(first_output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nAll public sample cases matched on important fields.")
    return 0


def _to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


if __name__ == "__main__":
    raise SystemExit(main())
