"""Minimal deployment smoke test; never prints request text."""

from __future__ import annotations

import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def call(url: str, payload: dict | None = None) -> tuple[int, dict]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST" if body else "GET")
    with urlopen(request, timeout=20) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    try:
        health_status, health = call(f"{base}/health")
        stance_status, stance = call(f"{base}/predict-stance", {"text": "SMOKE_TEST_INPUT", "target": "SMOKE_TEST_TARGET"})
    except (HTTPError, URLError, TimeoutError) as error:
        print(f"Smoke test failed: {type(error).__name__}")
        return 1
    print(f"health: HTTP {health_status}, status={health.get('status')}")
    print(f"predict-stance: HTTP {stance_status}, label={stance.get('label')}")
    return 0 if health_status == 200 and stance_status == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
