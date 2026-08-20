from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
from typing import Any
from urllib.parse import parse_qs, urlparse


DEMO_WARNING = (
    "FICTIONAL DEMO DATA. NOT DLD, NOT PORTAL DATA, AND NOT FOR REAL OWNER CALLS."
)
DEMO_AS_OF = "2026-08-20"

DEMO_FIXTURES: dict[tuple[str, str, str, str], dict[str, Any]] = {
    ("damac lagoons", "nice", "4", "townhouse"): {
        "actual_transactions": {
            "source": "DEMO ONLY - FICTIONAL TRANSACTION FIXTURE",
            "as_of": DEMO_AS_OF,
            "count": 9,
            "low_aed": 3_550_000,
            "high_aed": 3_950_000,
            "median_aed": 3_750_000,
        },
        "current_listings": {
            "source": "DEMO ONLY - FICTIONAL LISTING FIXTURE",
            "as_of": DEMO_AS_OF,
            "count": 12,
            "low_aed": 3_850_000,
            "high_aed": 4_200_000,
            "median_aed": 4_000_000,
        },
        "confidence": "high",
    },
    ("damac lagoons", "malta", "4", "townhouse"): {
        "actual_transactions": {
            "source": "DEMO ONLY - FICTIONAL TRANSACTION FIXTURE",
            "as_of": DEMO_AS_OF,
            "count": 6,
            "low_aed": 3_250_000,
            "high_aed": 3_700_000,
            "median_aed": 3_480_000,
        },
        "current_listings": {
            "source": "DEMO ONLY - FICTIONAL LISTING FIXTURE",
            "as_of": DEMO_AS_OF,
            "count": 8,
            "low_aed": 3_600_000,
            "high_aed": 4_050_000,
            "median_aed": 3_820_000,
        },
        "confidence": "medium",
    },
    ("damac lagoons", "venice", "6", "villa"): {
        "actual_transactions": {
            "source": "DEMO ONLY - FICTIONAL TRANSACTION FIXTURE",
            "as_of": DEMO_AS_OF,
            "count": 4,
            "low_aed": 6_900_000,
            "high_aed": 7_800_000,
            "median_aed": 7_350_000,
        },
        "current_listings": {
            "source": "DEMO ONLY - FICTIONAL LISTING FIXTURE",
            "as_of": DEMO_AS_OF,
            "count": 7,
            "low_aed": 7_500_000,
            "high_aed": 8_600_000,
            "median_aed": 8_050_000,
        },
        "confidence": "medium",
    },
}


class DemoMarketDataHandler(BaseHTTPRequestHandler):
    server_version = "SpeakingAgentDemoMarket/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {"status": "ok", "demo": True, "warning": DEMO_WARNING},
            )
            return
        if parsed.path != "/comparables":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        parameters = parse_qs(parsed.query, keep_blank_values=True)
        required = ("project", "cluster", "bedrooms", "property_type", "months")
        if any(
            len(parameters.get(name, ())) != 1 or not parameters[name][0].strip()
            for name in required
        ):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "exactly one non-empty value is required per query field"},
            )
            return
        try:
            months = int(parameters["months"][0])
        except ValueError:
            months = 0
        if months < 1 or months > 24:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "months must be between 1 and 24"},
            )
            return

        key = (
            _normalize(parameters["project"][0]),
            _normalize(parameters["cluster"][0]),
            _normalize_bedrooms(parameters["bedrooms"][0]),
            _normalize(parameters["property_type"][0]),
        )
        fixture = DEMO_FIXTURES.get(key)
        if fixture is None:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {
                    "demo": True,
                    "warning": DEMO_WARNING,
                    "error": "no fictional fixture for this comparable property",
                },
            )
            return
        self._send_json(
            HTTPStatus.OK,
            {
                **fixture,
                "demo": True,
                "warning": DEMO_WARNING,
                "query": {
                    "project": parameters["project"][0],
                    "cluster": parameters["cluster"][0],
                    "bedrooms": parameters["bedrooms"][0],
                    "property_type": parameters["property_type"][0],
                    "months": months,
                },
            },
        )

    def log_message(self, format: str, *args: Any) -> None:
        return None

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def create_server(port: int = 8765) -> ThreadingHTTPServer:
    if port < 0 or port > 65_535:
        raise ValueError("port must be between 0 and 65535")
    server = ThreadingHTTPServer(("127.0.0.1", port), DemoMarketDataHandler)
    server.daemon_threads = True
    return server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve clearly labeled fictional DAMAC market fixtures on loopback"
    )
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = create_server(args.port)
    host, port = server.server_address
    print(DEMO_WARNING, flush=True)
    print(f"Demo market endpoint: http://{host}:{port}/comparables", flush=True)
    print("Fixtures: Nice 4BR townhouse, Malta 4BR townhouse, Venice 6BR villa", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _normalize_bedrooms(value: str) -> str:
    match = re.search(r"\d+", value)
    return match.group(0) if match is not None else _normalize(value)


if __name__ == "__main__":
    raise SystemExit(main())