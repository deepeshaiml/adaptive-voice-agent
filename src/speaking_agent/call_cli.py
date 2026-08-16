from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
import json
import os
import sys
from uuid import uuid4

from speaking_agent.audio_recording import RecordingConsent
from speaking_agent.domain import ConversationContext
from speaking_agent.outbound import (
    allowed_test_numbers,
    ensure_controlled_test_number,
    mask_phone_number,
)


def _conversation_context(args: argparse.Namespace) -> ConversationContext:
    return ConversationContext(
        recipient_name=args.recipient_name,
        property_reference=args.property_reference,
        known_fields={
            name: value
            for name, value in (
                ("property_location", args.property_location),
                ("property_type", args.property_type),
            )
            if value is not None
        },
    )


def _dispatch_metadata(args: argparse.Namespace) -> dict[str, object]:
    context = _conversation_context(args)
    metadata: dict[str, object] = {"phone_number": args.phone_number}
    if context.recipient_name is not None or context.known_fields:
        conversation_context: dict[str, object] = {
            "known_fields": context.known_fields,
        }
        if context.recipient_name is not None:
            conversation_context.update(
                {
                    "recipient_name": context.recipient_name,
                    "property_reference": context.property_reference,
                }
            )
        metadata["conversation_context"] = conversation_context
    if args.recording_consent_reference is not None:
        consent = RecordingConsent(args.recording_consent_reference)
        metadata["recording_consent_reference"] = consent.reference
    return metadata


async def dispatch_call(args: argparse.Namespace) -> int:
    allowlist = allowed_test_numbers(
        os.environ.get("SPEAKING_AGENT_ALLOWED_TEST_NUMBERS")
    )
    ensure_controlled_test_number(args.phone_number, allowlist)
    context = _conversation_context(args)
    if args.recording_consent_reference is not None:
        RecordingConsent(args.recording_consent_reference)
    room_name = args.room or f"speaking-agent-test-{uuid4().hex[:12]}"
    summary = {
        "agent_name": args.agent_name,
        "room": room_name,
        "phone_number": mask_phone_number(args.phone_number),
        "execute": args.execute,
        "personalized": context.recipient_name is not None,
        "recording_consent_provided": args.recording_consent_reference is not None,
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
                metadata=json.dumps(_dispatch_metadata(args)),
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
    parser.add_argument("--recipient-name")
    parser.add_argument("--property-reference")
    parser.add_argument("--property-location")
    parser.add_argument("--property-type")
    parser.add_argument(
        "--recording-consent-reference",
        help="External per-call consent/audit reference for enabled recording",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(arguments)
    if bool(args.recipient_name) != bool(args.property_reference):
        parser.error(
            "--recipient-name and --property-reference must be provided together"
        )
    return args


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
