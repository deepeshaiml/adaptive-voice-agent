from __future__ import annotations

import asyncio
import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import json
import re
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


class LeadPriority(StrEnum):
    HOT = "PRIORITY_1_HOT"
    OPEN_TO_OFFER = "PRIORITY_2_OPEN_TO_OFFER"
    POTENTIAL = "PRIORITY_3_POTENTIAL"
    FUTURE = "PRIORITY_4_FUTURE"
    NOT_INTERESTED = "NOT_INTERESTED"
    UNQUALIFIED = "UNQUALIFIED"


class NotificationMode(StrEnum):
    IMMEDIATE = "IMMEDIATE"
    MARKET_FOLLOW_UP = "MARKET_FOLLOW_UP"
    NONE = "NONE"


class LeadDeliveryError(RuntimeError):
    """A completed lead could not be delivered to the configured workflow."""


@dataclass(frozen=True, slots=True)
class SalesCallAnalysis:
    priority: LeadPriority
    notification_mode: NotificationMode
    follow_up_at: str | None
    create_follow_up_task: bool
    recommended_next_action: str
    structured_summary: dict[str, Any]
    summary_text: str

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        summary_text: str,
    ) -> SalesCallAnalysis:
        try:
            return cls(
                priority=LeadPriority(data["priority"]),
                notification_mode=NotificationMode(data["notification_mode"]),
                follow_up_at=data.get("follow_up_at"),
                create_follow_up_task=bool(data["create_follow_up_task"]),
                recommended_next_action=str(data["recommended_next_action"]),
                structured_summary=dict(data["structured_summary"]),
                summary_text=summary_text,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Persisted sales call analysis is invalid") from error

    def as_dict(self) -> dict[str, Any]:
        return {
            "priority": self.priority.value,
            "notification_mode": self.notification_mode.value,
            "follow_up_at": self.follow_up_at,
            "create_follow_up_task": self.create_follow_up_task,
            "recommended_next_action": self.recommended_next_action,
            "structured_summary": self.structured_summary,
        }


@dataclass(frozen=True, slots=True)
class LeadWorkflowEvent:
    call_id: str
    campaign_id: str
    owner_name: str | None
    phone_number: str | None
    phone_number_masked: str | None
    analysis: SalesCallAnalysis
    transcript: tuple[dict[str, str], ...]
    recording_url: str | None = None

    def payload(self) -> dict[str, Any]:
        whatsapp = self.analysis.structured_summary.get("whatsapp", {})
        whatsapp_allowed = (
            isinstance(whatsapp, dict)
            and whatsapp.get("permission") == "Yes"
            and whatsapp.get("number_confirmed") == "Yes"
        )
        whatsapp_url = (
            build_whatsapp_url(self.phone_number, self._whatsapp_message())
            if self.phone_number is not None
            and whatsapp_allowed
            and self.analysis.priority
            not in {LeadPriority.NOT_INTERESTED, LeadPriority.UNQUALIFIED}
            else None
        )
        return {
            "event": "connected_call_analyzed",
            "call_id": self.call_id,
            "campaign_id": self.campaign_id,
            "owner_name": self.owner_name,
            "phone_number": self.phone_number,
            "phone_number_masked": self.phone_number_masked,
            "priority": self.analysis.priority.value,
            "notification_mode": self.analysis.notification_mode.value,
            "notify_yasir": self.analysis.notification_mode != NotificationMode.NONE,
            "notification_title": _notification_title(self.analysis.priority),
            "summary": self.analysis.summary_text,
            "structured_summary": self.analysis.structured_summary,
            "recommended_next_action": self.analysis.recommended_next_action,
            "open_whatsapp_url": whatsapp_url,
            "create_follow_up_task": self.analysis.create_follow_up_task,
            "follow_up_at": self.analysis.follow_up_at,
            "transcript": [dict(turn) for turn in self.transcript],
            "recording_url": self.recording_url,
        }

    def _whatsapp_message(self) -> str:
        property_data = self.analysis.structured_summary["property"]
        cluster = property_data.get("cluster") or "DAMAC Lagoons"
        return (
            f"Hello, this is Yasir following up on our call about your "
            f"{cluster} property."
        )


class LeadWorkflowSink(Protocol):
    async def publish(self, event: LeadWorkflowEvent) -> None: ...


class WebhookLeadWorkflowSink:
    def __init__(
        self,
        endpoint: str,
        *,
        bearer_token: str | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Lead workflow endpoint must be an HTTP(S) URL")
        if parsed.scheme != "https" and parsed.hostname not in {
            "127.0.0.1",
            "localhost",
        }:
            raise ValueError("Remote lead workflow endpoints must use HTTPS")
        if timeout_seconds <= 0:
            raise ValueError("Lead workflow timeout must be positive")
        self.endpoint = endpoint
        self.bearer_token = bearer_token
        self.timeout_seconds = timeout_seconds

    async def publish(self, event: LeadWorkflowEvent) -> None:
        await asyncio.to_thread(self._publish, event.payload())

    def _publish(self, payload: dict[str, Any]) -> None:
        headers = {"Content-Type": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response.read()
        except HTTPError as error:
            raise LeadDeliveryError(
                f"Lead workflow failed with HTTP {error.code}"
            ) from error
        except (TimeoutError, URLError, OSError) as error:
            raise LeadDeliveryError("Lead workflow request failed") from error


def analyze_sales_call(
    *,
    outcome: str,
    fields: dict[str, Any],
    transcript: tuple[dict[str, str], ...] = (),
    owner_name: str | None = None,
    phone_number_masked: str | None = None,
    completed_at: str | None = None,
    duration_seconds: float = 0.0,
    market_data: dict[str, Any] | None = None,
    market_feedback_discussed: bool = False,
    recording_url: str | None = None,
) -> SalesCallAnalysis:
    completed = _parse_datetime(completed_at) or datetime.now(timezone.utc)
    priority = _classify_priority(outcome, fields)
    notification_mode = _notification_mode(priority)
    follow_up_at = _follow_up_at(fields, completed)
    create_follow_up_task = priority in {
        LeadPriority.HOT,
        LeadPriority.OPEN_TO_OFFER,
        LeadPriority.POTENTIAL,
        LeadPriority.FUTURE,
    }
    recommended_action = _recommended_action(priority, fields, follow_up_at)
    structured = {
        "owner": {
            "name": owner_name,
            "phone": phone_number_masked,
        },
        "call": {
            "date": completed.date().isoformat(),
            "time": completed.time().replace(microsecond=0).isoformat(),
            "duration_seconds": round(duration_seconds, 2),
        },
        "property": {
            "project": fields.get("project"),
            "cluster": fields.get("cluster"),
            "bedrooms": fields.get("bedrooms"),
            "property_type": fields.get("property_type"),
            "unit_number": fields.get("unit_number"),
            "bua": fields.get("bua"),
            "plot_size": fields.get("plot_size"),
            "corner_or_middle": fields.get("unit_position"),
            "row_type": fields.get("row_type"),
            "view": fields.get("view"),
            "payment_status": fields.get("payment_status"),
            "handover": _join_values(
                fields.get("handover_status"),
                fields.get("handover_date"),
            ),
        },
        "seller_position": {
            "selling_intention": fields.get("selling_intention") or outcome,
            "asking_price": fields.get("asking_price"),
            "minimum_price_mentioned": fields.get("minimum_price"),
            "selling_timeline": fields.get("selling_timeline"),
        },
        "market_discussion": {
            "owner_asked_about_market_price": _owner_asked_about_market(transcript),
            "market_price_discussed": market_feedback_discussed,
            "recent_transactions_discussed": (
                market_feedback_discussed
                and bool((market_data or {}).get("recent_actual_transactions"))
            ),
            "current_listings_discussed": (
                market_feedback_discussed
                and bool((market_data or {}).get("current_asking_listings"))
            ),
            "evidence": market_data,
        },
        "whatsapp": {
            "permission": _yes_no(fields.get("whatsapp_permission")),
            "number_confirmed": _yes_no(
                fields.get("whatsapp_number_confirmed")
            ),
        },
        "documents_requested": {
            "floor_plan": _yes_no(fields.get("floor_plan_available")),
            "site_plan": _yes_no(fields.get("site_plan_available")),
            "payment_plan": _yes_no(fields.get("payment_plan_available")),
            "other": fields.get("other_documents"),
        },
        "follow_up": {
            "yasir_follow_up_required": create_follow_up_task,
            "priority": _priority_label(priority),
            "priority_code": priority.value,
            "follow_up_date": (
                follow_up_at[:10] if follow_up_at is not None else fields.get("follow_up_date")
            ),
            "follow_up_time": (
                follow_up_at[11:19]
                if follow_up_at is not None and "T" in follow_up_at
                else fields.get("follow_up_time")
            ),
            "requested_timing": fields.get("follow_up_timing"),
        },
        "owner_main_comments": _owner_comments(transcript),
        "objections_or_concerns": _objections(transcript),
        "recommended_next_action": recommended_action,
        "recording_url": recording_url,
    }
    return SalesCallAnalysis(
        priority=priority,
        notification_mode=notification_mode,
        follow_up_at=follow_up_at,
        create_follow_up_task=create_follow_up_task,
        recommended_next_action=recommended_action,
        structured_summary=structured,
        summary_text=format_call_summary(structured),
    )


def format_call_summary(summary: dict[str, Any]) -> str:
    owner = summary["owner"]
    call = summary["call"]
    property_data = summary["property"]
    seller = summary["seller_position"]
    market = summary["market_discussion"]
    whatsapp = summary["whatsapp"]
    documents = summary["documents_requested"]
    follow_up = summary["follow_up"]
    lines = [
        "CALL SUMMARY",
        "Owner",
        f"Name: {_display(owner.get('name'))}",
        f"Phone: {_display(owner.get('phone'))}",
        "",
        "Call",
        f"Date: {_display(call.get('date'))}",
        f"Time: {_display(call.get('time'))}",
        f"Duration: {_display(call.get('duration_seconds'))} seconds",
        "",
        "Property",
        f"Project: {_display(property_data.get('project'))}",
        f"Cluster: {_display(property_data.get('cluster'))}",
        f"Bedrooms: {_display(property_data.get('bedrooms'))}",
        f"Property Type: {_display(property_data.get('property_type'))}",
        f"Unit Number: {_display(property_data.get('unit_number'))}",
        f"BUA: {_display(property_data.get('bua'))}",
        f"Plot Size: {_display(property_data.get('plot_size'))}",
        f"Corner/Middle: {_display(property_data.get('corner_or_middle'))}",
        f"Single Row/Back-to-Back: {_display(property_data.get('row_type'))}",
        f"View: {_display(property_data.get('view'))}",
        f"Payment Status: {_display(property_data.get('payment_status'))}",
        f"Handover: {_display(property_data.get('handover'))}",
        "",
        "Seller Position",
        f"Selling Intention: {_display(seller.get('selling_intention'))}",
        f"Asking Price: {_display(seller.get('asking_price'))}",
        f"Minimum Price Mentioned: {_display(seller.get('minimum_price_mentioned'))}",
        f"Selling Timeline: {_display(seller.get('selling_timeline'))}",
        "",
        "Market Discussion",
        f"Owner Asked About Market Price: {_yes_no(market.get('owner_asked_about_market_price'))}",
        f"Market Price Discussed: {_yes_no(market.get('market_price_discussed'))}",
        f"Recent Transactions Discussed: {_yes_no(market.get('recent_transactions_discussed'))}",
        f"Current Listings Discussed: {_yes_no(market.get('current_listings_discussed'))}",
        "",
        "WhatsApp",
        f"Permission to WhatsApp: {_display(whatsapp.get('permission'))}",
        f"WhatsApp Number Confirmed: {_display(whatsapp.get('number_confirmed'))}",
        "",
        "Documents Requested",
        f"Floor Plan: {_display(documents.get('floor_plan'))}",
        f"Site Plan: {_display(documents.get('site_plan'))}",
        f"Payment Plan: {_display(documents.get('payment_plan'))}",
        f"Other: {_display(documents.get('other'))}",
        "",
        "Follow-Up",
        f"Yasir Follow-Up Required: {_yes_no(follow_up.get('yasir_follow_up_required'))}",
        f"Priority: {_display(follow_up.get('priority'))}",
        f"Follow-Up Date: {_display(follow_up.get('follow_up_date'))}",
        f"Follow-Up Time: {_display(follow_up.get('follow_up_time'))}",
        "",
        "Owner's Main Comments",
        _display(summary.get("owner_main_comments")),
        "",
        "Objections / Concerns",
        _display(summary.get("objections_or_concerns")),
        "",
        "Recommended Next Action",
        _display(summary.get("recommended_next_action")),
    ]
    return "\n".join(lines)


def build_whatsapp_url(phone_number: str, message: str = "") -> str:
    if re.fullmatch(r"\+[1-9][0-9]{7,14}", phone_number) is None:
        raise ValueError("WhatsApp number must use E.164 format")
    url = f"https://wa.me/{phone_number[1:]}"
    return f"{url}?text={quote(message)}" if message else url


def _classify_priority(outcome: str, fields: dict[str, Any]) -> LeadPriority:
    intention = str(fields.get("selling_intention", "")).casefold()
    if outcome == "SELL":
        if "selling now" in intention or intention in {"ready to sell", "sell now"}:
            return LeadPriority.HOT
        if "right price" in intention or "open to" in intention:
            return LeadPriority.OPEN_TO_OFFER
        if "later" in intention:
            return LeadPriority.FUTURE
        return LeadPriority.POTENTIAL
    if outcome == "FUTURE":
        return LeadPriority.FUTURE
    if outcome in {"NOT_INTERESTED", "DO_NOT_CONTACT", "WRONG_NUMBER"}:
        return LeadPriority.NOT_INTERESTED
    return LeadPriority.UNQUALIFIED


def _notification_mode(priority: LeadPriority) -> NotificationMode:
    if priority in {LeadPriority.HOT, LeadPriority.OPEN_TO_OFFER}:
        return NotificationMode.IMMEDIATE
    if priority == LeadPriority.POTENTIAL:
        return NotificationMode.MARKET_FOLLOW_UP
    return NotificationMode.NONE


def _notification_title(priority: LeadPriority) -> str | None:
    return {
        LeadPriority.HOT: "NEW DAMAC LAGOONS HOT SELLER",
        LeadPriority.OPEN_TO_OFFER: "NEW DAMAC LAGOONS OWNER OPEN TO OFFER",
        LeadPriority.POTENTIAL: "DAMAC LAGOONS MARKET FOLLOW-UP REQUIRED",
    }.get(priority)


def _follow_up_at(fields: dict[str, Any], now: datetime) -> str | None:
    date_value = str(fields.get("follow_up_date") or "").strip()
    time_value = str(fields.get("follow_up_time") or "").strip()
    if date_value:
        candidate = _parse_datetime(
            f"{date_value}T{time_value}" if time_value else date_value
        )
        if candidate is not None:
            if candidate.tzinfo is None:
                candidate = candidate.replace(tzinfo=now.tzinfo)
            return candidate.isoformat()

    timing = str(fields.get("follow_up_timing") or "").casefold()
    if "immediate" in timing or "asap" in timing or "right now" in timing:
        return now.isoformat()
    if "later today" in timing:
        return now.replace(hour=17, minute=0, second=0, microsecond=0).isoformat()
    if "tomorrow" in timing:
        return (now + timedelta(days=1)).replace(
            hour=10,
            minute=0,
            second=0,
            microsecond=0,
        ).isoformat()
    month_match = re.search(r"\b(3|6|12)\s*months?\b", timing)
    if month_match:
        return _add_months(now, int(month_match.group(1))).isoformat()
    if "handover" in timing:
        handover = _parse_datetime(str(fields.get("handover_date") or ""))
        if handover is not None:
            if handover.tzinfo is None:
                handover = handover.replace(tzinfo=now.tzinfo)
            return (handover - timedelta(days=30)).isoformat()
    return None


def _recommended_action(
    priority: LeadPriority,
    fields: dict[str, Any],
    follow_up_at: str | None,
) -> str:
    price = fields.get("minimum_price") or fields.get("asking_price")
    if priority in {LeadPriority.HOT, LeadPriority.OPEN_TO_OFFER}:
        timing = "immediately" if follow_up_at is None else f"at {follow_up_at}"
        price_clause = f" and confirm whether {price} is acceptable" if price else ""
        if fields.get("whatsapp_permission") is True:
            return (
                f"Yasir should WhatsApp the owner {timing}, request the floor plan, site "
                f"plan, payment statement, and exact unit location{price_clause}."
            )
        return (
            f"Yasir should follow up through the agreed channel {timing}{price_clause}; "
            "do not send a WhatsApp message without permission."
        )
    if priority == LeadPriority.POTENTIAL:
        return (
            "Yasir should send grounded transaction and listing comparables, then ask "
            "what price would make the owner sell."
        )
    if priority == LeadPriority.FUTURE:
        when = follow_up_at or "the owner's requested future date"
        return f"Create a follow-up task for {when} and revisit the sale closer to that date."
    if priority == LeadPriority.NOT_INTERESTED:
        return "No sales follow-up is required; preserve any do-not-contact instruction."
    return "Review the transcript before deciding whether any follow-up is appropriate."


def _owner_asked_about_market(transcript: tuple[dict[str, str], ...]) -> bool:
    for turn in transcript:
        if turn.get("role") != "owner":
            continue
        text = turn.get("text", "")
        is_question = "?" in text or re.match(
            r"^(?:what|how|can|could|would|is|are|do|does)\b",
            text.strip().casefold(),
        ) is not None
        if is_question and re.search(
            r"\b(?:worth|market|price|transaction|listing|comparable|valuation)\b",
            text.casefold(),
        ) is not None:
            return True
    return False


def _priority_label(priority: LeadPriority) -> str:
    return {
        LeadPriority.HOT: "Hot",
        LeadPriority.OPEN_TO_OFFER: "Hot - Open to Offer",
        LeadPriority.POTENTIAL: "Warm - Market Follow-Up",
        LeadPriority.FUTURE: "Future",
        LeadPriority.NOT_INTERESTED: "Not Interested",
        LeadPriority.UNQUALIFIED: "Unqualified",
    }[priority]


def _owner_comments(transcript: tuple[dict[str, str], ...]) -> str | None:
    comments = [
        " ".join(turn.get("text", "").split())
        for turn in transcript
        if turn.get("role") == "owner" and turn.get("text", "").strip()
    ]
    if not comments:
        return None
    combined = " ".join(comments[-6:])
    return combined if len(combined) <= 700 else f"{combined[:697].rstrip()}..."


def _objections(transcript: tuple[dict[str, str], ...]) -> str | None:
    owner_text = " ".join(
        turn.get("text", "")
        for turn in transcript
        if turn.get("role") == "owner"
    ).casefold()
    patterns = (
        (r"\b(?:already|have|using) (?:an? )?agent\b", "Already has an agent"),
        (r"\b(?:price|offer).{0,30}(?:low|too low)\b", "Price or offer is too low"),
        (r"\bwait.{0,30}handover\b|\bafter handover\b", "Waiting for handover"),
        (r"\b(?:not|don't|do not) (?:want to )?sell\b", "Does not want to sell now"),
        (r"\b(?:buyer first|have a buyer|actual buyer)\b", "Wants a buyer first"),
    )
    concerns = [label for pattern, label in patterns if re.search(pattern, owner_text)]
    return "; ".join(concerns) or None


def _add_months(value: datetime, months: int) -> datetime:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _yes_no(value: Any) -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "Not captured"


def _display(value: Any) -> str:
    if value in (None, ""):
        return "Not captured"
    return str(value)


def _join_values(*values: Any) -> str | None:
    present = [str(value) for value in values if value not in (None, "")]
    return "; ".join(present) or None