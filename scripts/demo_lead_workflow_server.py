from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from typing import Any
from urllib.parse import urlparse


MAX_EVENT_BYTES = 1_000_000
DEMO_WARNING = (
    "LOCAL DEMO RECEIVER. USE ONLY FAKE OWNERS AND CONTROLLED TEST NUMBERS."
)


class DemoLeadStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.is_symlink():
            raise ValueError("demo lead output cannot be a symbolic link")
        parent_existed = self.path.parent.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not parent_existed:
            self.path.parent.chmod(0o700)
        self._lock = threading.Lock()
        self._call_ids: set[str] = set()
        self._latest: dict[str, Any] | None = None
        self._latest_notification: dict[str, Any] | None = None
        self._load_existing()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._call_ids)

    @property
    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._latest) if self._latest is not None else None

    @property
    def latest_notification(self) -> dict[str, Any] | None:
        with self._lock:
            return (
                dict(self._latest_notification)
                if self._latest_notification is not None
                else None
            )

    def add(self, payload: dict[str, Any]) -> bool:
        call_id = str(payload["call_id"])
        with self._lock:
            if call_id in self._call_ids:
                return False
            record = {
                "received_at": datetime.now(timezone.utc).isoformat(),
                "payload": payload,
            }
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(self.path, flags, 0o600)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "a", encoding="utf-8") as output:
                json.dump(record, output, ensure_ascii=True, sort_keys=True)
                output.write("\n")
            self._call_ids.add(call_id)
            self._latest = dict(payload)
            if payload.get("notify_yasir") is True:
                self._latest_notification = dict(payload)
            return True

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        self.path.chmod(0o600)
        with self.path.open(encoding="utf-8") as existing:
            for line_number, line in enumerate(existing, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    payload = record["payload"]
                    call_id = payload["call_id"]
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise ValueError(
                        f"invalid demo lead record on line {line_number}"
                    ) from error
                if not isinstance(payload, dict) or not isinstance(call_id, str):
                    raise ValueError(
                        f"invalid demo lead record on line {line_number}"
                    )
                self._call_ids.add(call_id)
                self._latest = dict(payload)
                if payload.get("notify_yasir") is True:
                    self._latest_notification = dict(payload)


def create_server(
    output: str | Path,
    port: int = 8766,
) -> ThreadingHTTPServer:
    if port < 0 or port > 65_535:
        raise ValueError("port must be between 0 and 65535")
    store = DemoLeadStore(output)

    class DemoLeadWorkflowHandler(BaseHTTPRequestHandler):
        server_version = "SpeakingAgentDemoLeadWorkflow/1.0"

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "demo": True,
                        "events": store.count,
                        "warning": DEMO_WARNING,
                    },
                )
                return
            if path == "/latest":
                latest = store.latest
                if latest is None:
                    self._send_json(
                        HTTPStatus.NOT_FOUND,
                        {"error": "no demo lead has been received"},
                    )
                else:
                    self._send_json(HTTPStatus.OK, latest)
                return
            if path == "/latest-yasir":
                latest_notification = store.latest_notification
                if latest_notification is None:
                    self._send_json(
                        HTTPStatus.NOT_FOUND,
                        {"error": "no Yasir notification has been received"},
                    )
                else:
                    self._send_json(HTTPStatus.OK, latest_notification)
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/events":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            content_length = self.headers.get("Content-Length")
            try:
                body_size = int(content_length or "")
            except ValueError:
                body_size = 0
            if body_size < 1 or body_size > MAX_EVENT_BYTES:
                self._send_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": "event body must contain 1-1000000 bytes"},
                )
                return
            try:
                payload = json.loads(self.rfile.read(body_size))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "event body must be valid JSON"},
                )
                return
            if not _valid_event(payload):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "invalid connected_call_analyzed demo event"},
                )
                return
            added = store.add(payload)
            if added:
                _print_notification(payload)
            self._send_json(
                HTTPStatus.ACCEPTED if added else HTTPStatus.OK,
                {
                    "accepted": True,
                    "duplicate": not added,
                    "call_id": payload["call_id"],
                },
            )

        def log_message(self, format: str, *args: Any) -> None:
            return None

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode(
                "utf-8"
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", port), DemoLeadWorkflowHandler)
    server.daemon_threads = True
    return server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive and store fake local Yasir/CRM lead notifications"
    )
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/demo/lead_events.jsonl"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = create_server(args.output, args.port)
    host, port = server.server_address
    print(DEMO_WARNING, flush=True)
    print(f"Demo lead endpoint: http://{host}:{port}/events", flush=True)
    print(f"Latest event: http://{host}:{port}/latest", flush=True)
    print(f"Latest Yasir alert: http://{host}:{port}/latest-yasir", flush=True)
    print(f"Private JSONL output: {args.output}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _valid_event(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("event") == "connected_call_analyzed"
        and isinstance(payload.get("call_id"), str)
        and bool(payload["call_id"].strip())
        and isinstance(payload.get("campaign_id"), str)
        and bool(payload["campaign_id"].strip())
    )


def _print_notification(payload: dict[str, Any]) -> None:
    structured = payload.get("structured_summary")
    structured = structured if isinstance(structured, dict) else {}
    property_data = structured.get("property")
    property_data = property_data if isinstance(property_data, dict) else {}
    seller = structured.get("seller_position")
    seller = seller if isinstance(seller, dict) else {}
    notify_yasir = payload.get("notify_yasir") is True
    print(
        "\n=== DEMO YASIR NOTIFICATION ==="
        if notify_yasir
        else "\n=== DEMO CALL STORED - NO YASIR NOTIFICATION ===",
        flush=True,
    )
    print(
        payload.get("notification_title")
        or ("CALL REVIEW" if notify_yasir else "NO URGENT FOLLOW-UP"),
        flush=True,
    )
    print(f"Owner: {payload.get('owner_name') or 'Not captured'}", flush=True)
    print(f"Phone: {payload.get('phone_number_masked') or 'Not captured'}", flush=True)
    print(
        "Property: "
        f"{property_data.get('project') or 'Not captured'} / "
        f"{property_data.get('cluster') or 'Not captured'} / "
        f"{property_data.get('bedrooms') or '?'}BR "
        f"{property_data.get('property_type') or ''}".rstrip(),
        flush=True,
    )
    print(
        f"Intention: {seller.get('selling_intention') or 'Not captured'}",
        flush=True,
    )
    print(f"Asking: {seller.get('asking_price') or 'Not captured'}", flush=True)
    print(f"Priority: {payload.get('priority') or 'Unclassified'}", flush=True)
    print(
        f"Action: {payload.get('recommended_next_action') or 'Review the call'}",
        flush=True,
    )
    if payload.get("open_whatsapp_url"):
        print(f"Open WhatsApp: {payload['open_whatsapp_url']}", flush=True)
    print("================================\n", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())