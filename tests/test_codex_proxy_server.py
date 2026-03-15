import time
import unittest
from python.helpers.codex_proxy_server import normalize_openai_response

class TestCodexProxyServer(unittest.TestCase):
    def test_normalize_missing_created_at(self):
        # Case 1: missing created_at
        payload = {"id": "123", "object": "response"}
        normalized = normalize_openai_response(payload)
        self.assertIn("created_at", normalized)
        self.assertIsInstance(normalized["created_at"], int)
        self.assertEqual(normalized["status"], "completed")
        self.assertEqual(normalized["output"], [])

    def test_normalize_existing_created_at(self):
        # Case 2: existing created_at
        payload = {"id": "123", "object": "response", "created_at": 1000}
        normalized = normalize_openai_response(payload)
        self.assertEqual(normalized["created_at"], 1000)

    def test_normalize_minimal_completed_response(self):
        # Case 3: minimal completed response
        payload = {"status": "completed", "output": []}
        normalized = normalize_openai_response(payload)
        self.assertIn("created_at", normalized)
        self.assertEqual(normalized["status"], "completed")
        self.assertEqual(normalized["output"], [])

    def test_normalize_malformed_payload(self):
        # Case 4: malformed payload
        payload = {}
        normalized = normalize_openai_response(payload)
        self.assertIn("created_at", normalized)
        self.assertEqual(normalized["status"], "completed")
        self.assertEqual(normalized["output"], [])

if __name__ == "__main__":
    unittest.main()
