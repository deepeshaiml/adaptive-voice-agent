from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
import json
from pathlib import Path

from speaking_agent.adapters.storage.sqlite import SQLiteCallRepository


def _positive_seconds(value: str) -> float:
    seconds = float(value)
    if seconds <= 0:
        raise argparse.ArgumentTypeError("interval must be positive")
    return seconds


async def run(args: argparse.Namespace) -> int:
    repository = SQLiteCallRepository(
        args.database,
        legacy_retention_days=args.legacy_retention_days,
    )
    await repository.prepare()
    try:
        while True:
            purged_rows = await repository.purge_expired()
            print(json.dumps({"purged_rows": purged_rows}), flush=True)
            if args.once:
                return 0
            await asyncio.sleep(args.interval_seconds)
    finally:
        await repository.close()


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Periodically purge expired structured call and attempt rows"
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/speaking_agent.db"),
    )
    parser.add_argument(
        "--interval-seconds",
        type=_positive_seconds,
        default=3_600.0,
    )
    parser.add_argument("--legacy-retention-days", type=int)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(run(parse_args(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())