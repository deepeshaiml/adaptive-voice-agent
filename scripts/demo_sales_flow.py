from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone

from speaking_agent.lead_workflow import (
    LeadWorkflowEvent,
    WebhookLeadWorkflowSink,
    analyze_sales_call,
)
from speaking_agent.market_data import ComparableProperty, HttpMarketDataProvider


async def run(
    market_url: str,
    lead_url: str,
    *,
    call_id: str,
) -> None:
    market_provider = HttpMarketDataProvider(market_url)
    snapshot = await market_provider.get_comparables(
        ComparableProperty(
            project="DAMAC Lagoons",
            cluster="Nice",
            bedrooms="4",
            property_type="townhouse",
        )
    )
    if snapshot is None or not snapshot.demo:
        raise RuntimeError("The configured market endpoint did not return demo data")

    print("\n=== DEMO IN-CALL MARKET FEEDBACK ===")
    print(snapshot.spoken_feedback())
    print("====================================\n")

    transcript = (
        {
            "role": "owner",
            "text": "This is a fake demo. I am selling my 4 bedroom Nice townhouse now.",
        },
        {
            "role": "owner",
            "text": "I am asking AED 4 million and Yasir may WhatsApp this test number.",
        },
    )
    analysis = analyze_sales_call(
        outcome="SELL",
        fields={
            "selling_intention": "selling now",
            "project": "DAMAC Lagoons",
            "cluster": "Nice",
            "bedrooms": "4",
            "property_type": "townhouse",
            "unit_number": "DEMO-NICE-001",
            "plot_size": "2,300 sq ft (fictional)",
            "bua": "2,200 sq ft (fictional)",
            "unit_position": "corner",
            "row_type": "single row",
            "view": "lagoon view (fictional)",
            "payment_status": "70% paid (fictional)",
            "handover_status": "expected",
            "handover_date": "2027-06-01",
            "asking_price": "AED 4 million (fictional)",
            "minimum_price": "AED 3.9 million (fictional)",
            "selling_timeline": "immediately",
            "whatsapp_permission": True,
            "whatsapp_number_confirmed": True,
            "floor_plan_available": True,
            "site_plan_available": True,
            "payment_plan_available": True,
            "follow_up_timing": "immediately",
        },
        transcript=transcript,
        owner_name="Demo Ahmed",
        phone_number_masked="***0000",
        completed_at=datetime.now(timezone.utc).isoformat(),
        duration_seconds=125.0,
        market_data=snapshot.prompt_context(),
        market_feedback_discussed=True,
    )
    event = LeadWorkflowEvent(
        call_id=call_id,
        campaign_id="neoai-property-owner-qualification",
        owner_name="Demo Ahmed",
        phone_number="+971500000000",
        phone_number_masked="***0000",
        analysis=analysis,
        transcript=transcript,
    )
    await WebhookLeadWorkflowSink(lead_url).publish(event)
    print(f"Demo lead {call_id!r} sent to {lead_url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send one entirely fictional seller flow through local demo services"
    )
    parser.add_argument(
        "--market-url",
        default="http://127.0.0.1:8765/comparables",
    )
    parser.add_argument(
        "--lead-url",
        default="http://127.0.0.1:8766/events",
    )
    parser.add_argument(
        "--call-id",
        default="demo-nice-hot-seller-1",
        help="Change this value to create another demo notification",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        asyncio.run(
            run(
                args.market_url,
                args.lead_url,
                call_id=args.call_id,
            )
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Demo failed: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())