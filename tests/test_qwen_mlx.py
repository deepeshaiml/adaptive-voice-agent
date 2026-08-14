from pathlib import Path
import asyncio
from threading import Event
import unittest

from speaking_agent.adapters.llm.qwen_mlx import MlxLmBackend, QwenMlxConversationModel
from speaking_agent.campaign import load_campaign
from speaking_agent.domain import ConversationState
from speaking_agent.model import ConversationModelError


CAMPAIGN_PATH = Path(__file__).parents[1] / "campaigns" / "property_owner.json"


class FakeBackend:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = None
        self.max_tokens = None

    async def prepare(self):
        return None

    async def close(self):
        return None

    async def generate(self, messages, *, max_tokens):
        self.messages = messages
        self.max_tokens = max_tokens
        return self.response


class QwenMlxConversationModelTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.campaign = load_campaign(CAMPAIGN_PATH)
        self.state = ConversationState(
            call_id="call-1",
            session_id="session-1",
            campaign_id=self.campaign.campaign_id,
            last_asked_field="intent",
        )

    async def test_translates_structured_model_output(self) -> None:
        backend = FakeBackend(
            '{"suggested_outcome":"SELL","field_updates":{"intent":"SELL",'
            '"selling_timeline":"two months"},"answer":null,'
            '"acknowledgement":"That sounds good.",'
            '"callback_requested":false,"human_transfer_requested":false}'
        )
        model = QwenMlxConversationModel(backend, max_tokens=200)

        interpretation = await model.interpret(
            "I might sell in two months",
            self.state,
            self.campaign,
        )

        self.assertEqual(interpretation.suggested_outcome, "SELL")
        self.assertEqual(interpretation.field_updates["selling_timeline"], "two months")
        self.assertEqual(interpretation.acknowledgement, "That sounds good.")
        self.assertEqual(backend.max_tokens, 200)
        self.assertIn("latest_owner_utterance", backend.messages[1]["content"])
        self.assertIn("outcome_guidance", backend.messages[1]["content"])
        self.assertIn("classification_rules", backend.messages[1]["content"])
        self.assertIn("fields_by_outcome", backend.messages[1]["content"])
        self.assertIn("field_allowed_values", backend.messages[1]["content"])
        self.assertIn("conversation_brief", backend.messages[1]["content"])
        self.assertIn("scenario_playbook", backend.messages[1]["content"])
        self.assertNotIn("call-1", backend.messages[1]["content"])
        self.assertIn("Never ignore a direct question", backend.messages[0]["content"])
        self.assertIn("flexible guidance, not a script", backend.messages[0]["content"])
        self.assertIn("including one-word answers", backend.messages[0]["content"])

    async def test_accepts_a_single_json_code_fence(self) -> None:
        backend = FakeBackend(
            "```json\n{\"suggested_outcome\":null,\"field_updates\":{}}\n```"
        )
        model = QwenMlxConversationModel(backend)

        interpretation = await model.interpret("Maybe", self.state, self.campaign)

        self.assertIsNone(interpretation.suggested_outcome)

    async def test_rejects_malformed_model_output(self) -> None:
        model = QwenMlxConversationModel(FakeBackend("not json"))

        with self.assertRaisesRegex(ConversationModelError, "invalid JSON"):
            await model.interpret("Maybe", self.state, self.campaign)

    async def test_stalled_native_generation_is_bounded_and_quarantined(self) -> None:
        started = Event()
        release = Event()

        class BlockingBackend(MlxLmBackend):
            def _generate_sync(self, messages, max_tokens, cancellation):
                del messages, max_tokens, cancellation
                started.set()
                release.wait()
                return "{}"

        backend = BlockingBackend(cancellation_grace_seconds=0.01)
        task = asyncio.create_task(backend.generate([], max_tokens=1))
        await asyncio.to_thread(started.wait, 1)

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.2)
        with self.assertRaisesRegex(ConversationModelError, "quarantined"):
            await backend.generate([], max_tokens=1)

        release.set()


if __name__ == "__main__":
    unittest.main()