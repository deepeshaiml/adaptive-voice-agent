import json
from pathlib import Path
import stat
import tempfile
import threading
import unittest
from urllib.request import urlopen

from scripts.demo_lead_workflow_server import create_server as create_lead_server
from scripts.demo_market_data_server import create_server as create_market_server
from speaking_agent.lead_workflow import (
    LeadWorkflowEvent,
    WebhookLeadWorkflowSink,
    analyze_sales_call,
)
from speaking_agent.market_data import ComparableProperty, HttpMarketDataProvider


class DemoIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_market_server_uses_real_provider_and_rejects_unknown_fixture(
        self,
    ) -> None:
        server = create_market_server(0)
        thread = _serve(server)
        provider = HttpMarketDataProvider(
            f"http://127.0.0.1:{server.server_port}/comparables"
        )
        try:
            snapshot = await provider.get_comparables(
                ComparableProperty(
                    "DAMAC Lagoons",
                    "Nice",
                    "4BR",
                    "townhouse",
                )
            )
            missing = await provider.get_comparables(
                ComparableProperty(
                    "DAMAC Lagoons",
                    "Morocco",
                    "5",
                    "villa",
                )
            )
        finally:
            _stop(server, thread)

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertTrue(snapshot.demo)
        self.assertIn("DEMO ONLY", snapshot.actual_transactions.source)
        self.assertTrue(snapshot.spoken_feedback().startswith("FICTIONAL DEMO DATA"))
        self.assertIsNone(missing)

    async def test_lead_receiver_uses_real_sink_and_deduplicates_call_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "lead_events.jsonl"
            server = create_lead_server(output, 0)
            thread = _serve(server)
            sink = WebhookLeadWorkflowSink(
                f"http://127.0.0.1:{server.server_port}/events"
            )
            analysis = analyze_sales_call(
                outcome="SELL",
                fields={
                    "selling_intention": "selling now",
                    "project": "DAMAC Lagoons",
                    "cluster": "Nice",
                    "bedrooms": "4",
                    "property_type": "townhouse",
                    "asking_price": "AED 4 million",
                    "whatsapp_permission": True,
                    "whatsapp_number_confirmed": True,
                },
                owner_name="Demo Ahmed",
                phone_number_masked="***4567",
            )
            event = LeadWorkflowEvent(
                call_id="demo-call-1",
                campaign_id="neoai-property-owner-qualification",
                owner_name="Demo Ahmed",
                phone_number="+971500000000",
                phone_number_masked="***0000",
                analysis=analysis,
                transcript=(
                    {"role": "owner", "text": "This is a fake demo call."},
                ),
            )
            try:
                await sink.publish(event)
                await sink.publish(event)
                unqualified = analyze_sales_call(
                    outcome="UNKNOWN",
                    fields={"intent": "UNKNOWN"},
                    owner_name="Demo Unknown",
                    phone_number_masked="***0000",
                )
                await sink.publish(
                    LeadWorkflowEvent(
                        call_id="demo-call-unknown",
                        campaign_id="neoai-property-owner-qualification",
                        owner_name="Demo Unknown",
                        phone_number="+971500000000",
                        phone_number_masked="***0000",
                        analysis=unqualified,
                        transcript=(),
                    )
                )
                with urlopen(
                    f"http://127.0.0.1:{server.server_port}/latest-yasir"
                ) as response:
                    latest_yasir = json.load(response)
            finally:
                _stop(server, thread)

            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            mode = stat.S_IMODE(output.stat().st_mode)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["payload"]["call_id"], "demo-call-1")
        self.assertIn("wa.me/971500000000", records[0]["payload"]["open_whatsapp_url"])
        self.assertEqual(latest_yasir["call_id"], "demo-call-1")
        self.assertEqual(mode, 0o600)


def _serve(server) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def _stop(server, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()