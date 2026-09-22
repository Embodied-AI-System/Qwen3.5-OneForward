import unittest
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from app import create_app
from history import HistoryStore
from inference import BatchResult, DecisionResult


class FakeBackend:
    def __init__(self):
        self.last_media = ()
        self.batch_calls = 0

    def prepare_media(self, items):
        return tuple({"type": item.type, "name": item.name} for item in items)

    def score_batch(self, state, jobs, *, media=()):
        self.batch_calls += 1
        self.last_media = media
        decisions = []
        for job in jobs:
            count = len(job["criteria"])
            probabilities = (0.1, 0.8, 0.1) if count == 3 else (0.2, 0.8)
            decisions.append(DecisionResult(
                probabilities=probabilities,
                selected_index=1,
                confidence=0.42,
                candidate_mass=0.9,
                raw_logits=tuple(float(index) for index in range(count)),
                candidate_tokens=tuple("ABCDEFGH"[:count]),
                candidate_token_ids=tuple(range(32, 32 + count)),
                prompt="rendered prompt",
                input_tokens=23,
                inference_ms=12.5,
                top_vocabulary=(),
            ))
        return BatchResult(
            decisions=tuple(decisions),
            batch_size=len(decisions),
            inference_ms=12.5,
            queue_ms=0.2,
            preprocess_ms=1.5,
            shared_prefix_tokens=19,
            max_input_tokens=23,
            total_input_tokens=23 * len(decisions),
            padded_input_tokens=23 * len(decisions),
            padding_tokens=0,
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
        self.temporary_directory = TemporaryDirectory()
        self.history = HistoryStore(
            f"{self.temporary_directory.name}/history.sqlite3"
        )
        self.client_context = TestClient(create_app(self.backend, self.history))
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.temporary_directory.cleanup()

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
        self.assertEqual(payload["_compat"]["question_execution"], "shared_prefix_batch")
        self.assertEqual(payload["_debug"]["batch"]["model_forward_count"], 1)
        self.assertFalse(payload["_compat"]["semantic_compatibility"])
        self.assertEqual(
            payload["answers"]["department"]["_debug"]["candidate_mass"],
            0.9,
        )

    def test_multiple_questions_use_one_batch_and_are_persisted(self):
        response = self.client.post(
            "/v1/systemone",
            json={
                "state": "shared state",
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "route it",
                        "criteria": {"billing": "pay", "technical": "bug", "sales": "buy"},
                    },
                    "priority": {
                        "type": "choice",
                        "instructions": "prioritize it",
                        "criteria": {"normal": "later", "urgent": "now"},
                    },
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(self.backend.batch_calls, 1)
        self.assertEqual(payload["_debug"]["batch"]["batch_size"], 2)
        self.assertEqual(payload["_debug"]["batch"]["shared_prefix_tokens"], 19)
        history = self.client.get("/v1/history").json()
        self.assertEqual(history["total"], 1)
        detail = self.client.get(f"/v1/history/{payload['id']}").json()
        self.assertEqual(detail["request"]["state"], "shared state")
        self.assertEqual(detail["metrics"]["model_forward_count"], 1)

    def test_history_entry_can_be_deleted(self):
        response = self.client.post(
            "/v1/systemone",
            json={
                "state": "delete me",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "pick",
                        "criteria": {"keep": "keep", "drop": "drop"},
                    }
                },
            },
        )
        entry_id = response.json()["id"]
        self.assertEqual(self.client.delete(f"/v1/history/{entry_id}").status_code, 204)
        self.assertEqual(self.client.get(f"/v1/history/{entry_id}").status_code, 404)

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
        detail = self.client.get(f"/v1/history/{response.json()['id']}").json()
        self.assertFalse(detail["request"]["media"][0]["content_stored"])
        self.assertNotIn("data_url", detail["request"]["media"][0])

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
