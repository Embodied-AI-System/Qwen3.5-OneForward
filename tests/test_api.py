import unittest

from fastapi.testclient import TestClient

from app import create_app
from inference import DecisionResult


class FakeBackend:
    def __init__(self):
        self.last_media = ()

    def prepare_media(self, items):
        return tuple({"type": item.type, "name": item.name} for item in items)

    def score_choice(self, state, instructions, criteria, media=()):
        self.last_media = media
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
        self.backend = FakeBackend()
        self.client_context = TestClient(create_app(self.backend))
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

    def test_multimodal_media_is_prepared_and_forwarded(self):
        response = self.client.post(
            "/v1/systemone",
            json={
                "state": "inspect",
                "media": [{
                    "type": "image",
                    "name": "pixel.png",
                    "data_url": "data:image/png;base64,AAAAAAAA",
                }],
                "questions": {
                    "decision": {
                        "type": "choice",
                        "instructions": "pick",
                        "criteria": {"red": "red", "blue": "blue"},
                    }
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.backend.last_media[0]["type"], "image")
        self.assertEqual(response.json()["_debug"]["media_count"], 1)

    def test_rejects_more_than_one_video(self):
        response = self.client.post(
            "/v1/systemone",
            json={
                "state": "inspect",
                "media": [
                    {"type": "video", "data_url": "data:video/mp4;base64,AAAAAAAA"},
                    {"type": "video", "data_url": "data:video/mp4;base64,AAAAAAAA"},
                ],
                "questions": {
                    "decision": {
                        "type": "choice",
                        "instructions": "pick",
                        "criteria": {"a": "a", "b": "b"},
                    }
                },
            },
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
