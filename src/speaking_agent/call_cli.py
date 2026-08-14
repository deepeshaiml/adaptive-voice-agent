from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
import json
import os
import sys
from uuid import uuid4

from speaking_agent.outbound import (
    allowed_test_numbers,
    ensure_controlled_test_number,
    mask_phone_number,
)


async def dispatch_call(args: argparse.Namespace) -> int:
    allowlist = allowed_test_numbers(
        os.environ.get("SPEAKING_AGENT_ALLOWED_TEST_NUMBERS")
    )
    ensure_controlled_test_number(args.phone_number, allowlist)
    room_name = args.room or f"speaking-agent-test-{uuid4().hex[:12]}"
    summary = {
        "agent_name": args.agent_name,
        "room": room_name,
        "phone_number": mask_phone_number(args.phone_number),
        "execute": args.execute,
    }
    if not args.execute:
        print(json.dumps(summary, indent=2))
        print("Dry run only. Pass --execute to dispatch the controlled test call.")
        return 0

    from livekit import api

    async with api.LiveKitAPI() as livekit_api:
        dispatch = await livekit_api.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=args.agent_name,
                room=room_name,
                metadata=json.dumps({"phone_number": args.phone_number}),
            )
        )
    summary["dispatch_id"] = dispatch.id
    print(json.dumps(summary, indent=2))
    return 0


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dispatch one explicitly allowlisted controlled test call"
    )
    parser.add_argument("phone_number")
    parser.add_argument("--agent-name", default="speaking-agent")
    parser.add_argument("--room")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        return asyncio.run(dispatch_call(parse_args(arguments)))
    except (PermissionError, ValueError) as error:
        print(f"Call blocked: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
