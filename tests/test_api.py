import unittest

from fastapi.testclient import TestClient

from app import create_app
from inference import DecisionResult


class FakeBackend:
    def score_choice(self, state, instructions, criteria):
        return DecisionResult(
            probabilities=(0.1, 0.8, 0.1),
            selected_index=1,
            confidence=0.42,
            candidate_mass=0.9,
            raw_logits=(1.0, 3.0, 1.0),
            candidate_tokens=("A", "B", "C"),
            candidate_token_ids=(32, 33, 34),
            prompt="rendered prompt",
            input_tokens=23,
            inference_ms=12.5,
            top_vocabulary=(),
        )

    def status(self):
        return {
            "status": "ready",
            "model": "fake-logits",
            "prompt_mode": "chat",
        }


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client_context = TestClient(create_app(FakeBackend()))
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def test_choice_response_shape(self):
        response = self.client.post(
            "/v1/systemone",
            json={
                "state": "integration fails",
                "model": "jev-latest",
                "questions": {
                    "department": {
                        "type": "choice",
                        "instructions": "route it",
                        "criteria": {
                            "billing": "payments",
                            "technical": "bugs",
                            "sales": "pricing",
                        },
                    }
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answers"]["department"]["choice"], "technical")
        self.assertEqual(payload["usage"], {"input_tokens": 23, "output_tokens": 0})
        self.assertFalse(payload["_compat"]["semantic_compatibility"])
        self.assertEqual(
            payload["answers"]["department"]["_debug"]["candidate_mass"],
            0.9,
        )

    def test_rejects_non_choice_question(self):
        response = self.client.post(
            "/v1/systemone",
            json={
                "state": "x",
                "questions": {
                    "urgent": {"type": "noul", "instructions": "urgent?"}
                },
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_requires_two_to_eight_options(self):
        response = self.client.post(
            "/v1/systemone",
            json={
                "state": "x",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "q",
                        "criteria": {"only": "one"},
                    }
                },
            },
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
