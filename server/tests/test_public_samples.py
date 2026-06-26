import json
import unittest
from pathlib import Path

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


class PublicSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))

    def test_public_samples_match_important_fields(self):
        for case in self.data["cases"]:
            with self.subTest(case=case["id"]):
                response = analyze_ticket(AnalyzeTicketRequest(**case["input"]))
                actual = _to_dict(response)
                expected = case["expected_output"]
                for field in IMPORTANT_FIELDS:
                    self.assertEqual(actual[field], expected[field], field)


def _to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


if __name__ == "__main__":
    unittest.main()

