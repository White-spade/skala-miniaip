import json
import unittest

from step22_grounding import source_node, verify_draft, validate_node


class GroundingTest(unittest.TestCase):
    def setUp(self):
        self.sources = source_node({"collected": [
            {"source": "monitoring", "target": "payments-service", "data": {
                "status": "degraded", "cpu_percent": 98.8,
                "memory_percent": 58.9, "alerts": ["CPU 91% (threshold 80%)"],
            }},
            {"source": "runbook", "target": "payments-service", "data":
                "# CPU 대응\n1. 최근 배포 이력을 확인한다.\n2. CPU 부하 원인을 조사한다."},
        ]})["sources"]
        self.valid = {
            "facts": [
                {"source_id": "S1", "field": "status", "value": "degraded"},
                {"source_id": "S1", "field": "cpu_percent", "value": 98.8},
                {"source_id": "S1", "field": "memory_percent", "value": 58.9},
            ],
            "steps": [{"source_id": "S2", "quote": "최근 배포 이력을 확인한다."}],
        }

    def test_valid(self):
        self.assertEqual(verify_draft(self.valid, self.sources), [])
        result = validate_node({"result": json.dumps(self.valid), "sources": self.sources})
        self.assertTrue(result["validation_passed"])
        self.assertIn("CPU 98.8%", result["result"])

    def test_wrong_number(self):
        bad = json.loads(json.dumps(self.valid))
        bad["facts"][1]["value"] = 999
        self.assertTrue(verify_draft(bad, self.sources))

    def test_fabricated_runbook_quote(self):
        bad = json.loads(json.dumps(self.valid))
        bad["steps"][0]["quote"] = "서버를 즉시 종료한다."
        self.assertTrue(verify_draft(bad, self.sources))

    def test_fallback_after_three_rewrites(self):
        bad = {"facts": [], "steps": []}
        for i in range(4):
            result = validate_node({
                "result": json.dumps(bad), "sources": self.sources,
                "validation_count": i,
            })
            self.assertEqual(result["validation_route"], "end" if i == 3 else "write")
        self.assertFalse(result["validation_passed"])
        self.assertIn("CPU 98.8%", result["result"])
        self.assertIn("MCP 원본 자료", result["result"])


if __name__ == "__main__":
    unittest.main()
