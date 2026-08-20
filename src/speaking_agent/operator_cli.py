from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import asdict
import json
from pathlib import Path

from speaking_agent.adapters.storage.sqlite import SQLiteCallRepository
from speaking_agent.metrics import aggregate_call_metrics


async def run(args: argparse.Namespace) -> int:
    repository = SQLiteCallRepository(
        args.database,
        legacy_retention_days=args.legacy_retention_days,
    )
    await repository.prepare()
    try:
        if args.command == "show":
            record = await repository.get(args.call_id)
            if record is None:
                print(json.dumps({"error": "call not found", "call_id": args.call_id}))
                return 1
            print(json.dumps(asdict(record), indent=2, sort_keys=True))
            return 0

        if args.command == "metrics":
            records = await repository.list_recent(args.limit)
            print(
                json.dumps(
                    asdict(aggregate_call_metrics(records)),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        records = await repository.list_recent(args.limit)
        summaries = [
            {
                "call_id": record.call_id,
                "completed_at": record.completed_at,
                "campaign_id": record.campaign_id,
                "connection_result": record.connection_result,
                "outcome": record.outcome,
                "qualified": record.qualified,
                "priority": record.priority,
                "follow_up_at": record.follow_up_at,
                "phone_number": record.phone_number_masked,
                "disconnected": record.disconnected,
                "duration_seconds": record.duration_seconds,
                "error": record.error,
            }
            for record in records
        ]
        print(json.dumps(summaries, indent=2, sort_keys=True))
        return 0
    finally:
        await repository.close()


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect persisted call outcomes")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/speaking_agent.db"),
    )
    parser.add_argument("--legacy-retention-days", type=int)
    commands = parser.add_subparsers(dest="command", required=True)
    list_parser = commands.add_parser("list")
    list_parser.add_argument("--limit", type=int, default=50)
    show_parser = commands.add_parser("show")
    show_parser.add_argument("call_id")
    metrics_parser = commands.add_parser("metrics")
    metrics_parser.add_argument("--limit", type=int, default=1_000)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(run(parse_args(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
