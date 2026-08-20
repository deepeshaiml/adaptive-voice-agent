from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Sequence

from speaking_agent.campaign import load_campaign
from speaking_agent.conversation import ConversationSession
from speaking_agent.mock_model import MockConversationModel
from speaking_agent.model import ConversationModel, ConversationModelError


async def run_simulator(campaign_path: Path, model: ConversationModel) -> int:
    try:
        await model.prepare()
        campaign = load_campaign(campaign_path)
        session = ConversationSession(campaign, model)
        print(f"Agent: {session.start().text}")

        while not session.state.ended:
            try:
                utterance = await asyncio.to_thread(input, "Owner: ")
            except EOFError:
                print()
                return 1
            if utterance.strip() == "/quit":
                return 1
            if not utterance.strip():
                continue

            reply = await session.receive(utterance)
            print(f"Agent: {reply.text}")

        print("Result:")
        print(json.dumps(asdict(session.result()), indent=2))
        return 0
    finally:
        await model.close()


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a text-only conversation simulation")
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path("campaigns/neoai_property_owner.json"),
        help="Path to a campaign JSON file",
    )
    parser.add_argument(
        "--model",
        choices=("mock", "mlx"),
        default="mock",
        help="Conversation model adapter",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local path or Hugging Face repository for the MLX model",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    model: ConversationModel
    if args.model == "mlx":
        from speaking_agent.adapters.llm.qwen_mlx import (
            DEFAULT_MODEL_PATH,
            MlxLmBackend,
            QwenMlxConversationModel,
        )

        model = QwenMlxConversationModel(
            MlxLmBackend(args.model_path or DEFAULT_MODEL_PATH)
        )
    else:
        model = MockConversationModel()
    try:
        return asyncio.run(run_simulator(args.campaign, model))
    except ConversationModelError as error:
        print(f"Model error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
