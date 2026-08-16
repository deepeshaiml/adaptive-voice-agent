from pathlib import Path
import asyncio
from dataclasses import replace
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
            recent_owner_utterances=[
                "I own an apartment in Dubai Marina.",
                "I might consider selling next year.",
            ],
            recent_dialogue=[
                {"role": "agent", "text": "Would you consider selling?"},
                {"role": "owner", "text": "Maybe next year."},
            ],
        )

    async def test_translates_structured_model_output(self) -> None:
        backend = FakeBackend(
            '{"suggested_outcome":"SELL","field_updates":{"intent":"SELL",'
            '"selling_timeline":"two months"},"answer":null,'
            '"acknowledgement":"That sounds good.",'
            '"next_question_field":"property_location",'
            '"next_question":"Which area is it in?",'
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
        self.assertEqual(interpretation.next_question_field, "property_location")
        self.assertEqual(interpretation.next_question, "Which area is it in?")
        self.assertEqual(backend.max_tokens, 200)
        self.assertIn("latest_owner_utterance", backend.messages[1]["content"])
        self.assertIn("outcome_guidance", backend.messages[1]["content"])
        self.assertIn("classification_rules", backend.messages[1]["content"])
        self.assertIn("fields_by_outcome", backend.messages[1]["content"])
        self.assertIn("field_allowed_values", backend.messages[1]["content"])
        self.assertIn("conversation_brief", backend.messages[1]["content"])
        self.assertIn("relevant_scenarios", backend.messages[1]["content"])
        self.assertIn("voice_style", backend.messages[1]["content"])
        self.assertIn("current_flow", backend.messages[1]["content"])
        self.assertIn("recent_dialogue", backend.messages[1]["content"])
        self.assertIn("Would you consider selling", backend.messages[1]["content"])
        self.assertIn("asked_field_counts", backend.messages[1]["content"])
        self.assertNotIn("call-1", backend.messages[1]["content"])
        self.assertIn("Never ignore a direct question", backend.messages[0]["content"])
        self.assertIn("flexible guidance, not a script", backend.messages[0]["content"])
        self.assertIn("one-word answers", backend.messages[0]["content"])
        self.assertIn("Keep output sparse", backend.messages[0]["content"])
        self.assertLess(
            sum(len(message["content"]) for message in backend.messages),
            15_000,
        )

    async def test_selects_relevant_scenarios_without_full_playbook(self) -> None:
        backend = FakeBackend('{"suggested_outcome":null,"field_updates":{}}')
        model = QwenMlxConversationModel(backend)

        await model.interpret(
            "How did you get my number?",
            self.state,
            self.campaign,
        )

        context = backend.messages[1]["content"]
        self.assertIn("contact-list FAQ", context)
        self.assertNotIn("owner becomes annoyed", context)

    def test_prompt_budget_trims_oldest_dialogue_without_mutating_state(self) -> None:
        self.state.recent_dialogue = [
            {
                "role": "owner" if index % 2 else "agent",
                "text": " ".join([f"turn-{index}"] * 20),
                "delivery": "delivered",
            }
            for index in range(25)
        ]
        original_dialogue = [dict(turn) for turn in self.state.recent_dialogue]

        messages = QwenMlxConversationModel._build_messages(
            "What did we discuss?",
            self.state,
            self.campaign,
        )

        prompt_size = sum(len(message["content"]) for message in messages)
        self.assertLessEqual(prompt_size, QwenMlxConversationModel.MAX_PROMPT_CHARS)
        self.assertEqual(self.state.recent_dialogue, original_dialogue)
        self.assertIn("turn-24", messages[1]["content"])
        self.assertNotIn("turn-0", messages[1]["content"])

    def test_irreducible_required_context_overflow_fails_clearly(self) -> None:
        oversized_campaign = replace(
            self.campaign,
            objective="required-objective " * QwenMlxConversationModel.MAX_PROMPT_CHARS,
        )

        with self.assertRaisesRegex(ConversationModelError, "prompt budget"):
            QwenMlxConversationModel._build_messages(
                "Keep this latest utterance",
                self.state,
                oversized_campaign,
            )

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