#!/usr/bin/env python3
"""Exercise the live endpoint and print the research readout."""

import json
import os
import urllib.request


base_url = os.environ.get("JEV_BASE_URL", "http://127.0.0.1:8000")
payload = {
    "state": "Customer says the integration keeps failing. Please help ASAP.",
    "model": "jev-latest",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Payment or subscription issues",
                "technical": "Bugs or integration problems",
                "sales": "Pricing or account questions",
            },
        }
    },
}
request = urllib.request.Request(
    f"{base_url}/v1/systemone",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=120) as response:
    print(json.dumps(json.load(response), indent=2, ensure_ascii=False))
