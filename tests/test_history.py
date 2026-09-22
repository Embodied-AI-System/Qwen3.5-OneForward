import unittest
from tempfile import TemporaryDirectory

from history import HistoryStore


class HistoryStoreTests(unittest.TestCase):
    def test_retention_keeps_latest_entries(self):
        with TemporaryDirectory() as directory:
            store = HistoryStore(f"{directory}/history.sqlite3", retention=2)
            ids = []
            for index in range(3):
                entry_id = store.begin({
                    "state": f"request {index}",
                    "questions": {"q": {"type": "choice"}},
                })
                store.fail(entry_id, {"detail": "boom"}, {"request_ms": index})
                ids.append(entry_id)
            self.assertEqual(store.list()["total"], 2)
            self.assertIsNone(store.get(ids[0]))
            self.assertEqual(store.get(ids[-1])["error"]["detail"], "boom")


if __name__ == "__main__":
    unittest.main()
