from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


class MarketDataError(RuntimeError):
    """Approved market data could not be retrieved or validated."""


@dataclass(frozen=True, slots=True)
class ComparableProperty:
    project: str
    cluster: str
    bedrooms: str
    property_type: str
    months: int = 6

    def __post_init__(self) -> None:
        for name in ("project", "cluster", "bedrooms", "property_type"):
            value = " ".join(getattr(self, name).split())
            if not value or len(value) > 120:
                raise ValueError(f"Comparable property {name} is invalid")
            object.__setattr__(self, name, value)
        if self.months < 1 or self.months > 24:
            raise ValueError("Comparable period must be between 1 and 24 months")


@dataclass(frozen=True, slots=True)
class PriceEvidence:
    source: str
    as_of: str
    count: int
    low_aed: int
    high_aed: int
    median_aed: int | None = None

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.as_of.strip():
            raise ValueError("Market evidence requires source and as_of")
        if self.count < 1:
            raise ValueError("Market evidence count must be positive")
        if self.low_aed < 1 or self.high_aed < self.low_aed:
            raise ValueError("Market evidence price range is invalid")
        if self.median_aed is not None and not (
            self.low_aed <= self.median_aed <= self.high_aed
        ):
            raise ValueError("Market evidence median must be within its range")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PriceEvidence:
        try:
            return cls(
                source=str(data["source"]),
                as_of=str(data["as_of"]),
                count=int(data["count"]),
                low_aed=int(data["low_aed"]),
                high_aed=int(data["high_aed"]),
                median_aed=(
                    int(data["median_aed"])
                    if data.get("median_aed") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise MarketDataError("Market evidence payload is invalid") from error


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    query: ComparableProperty
    actual_transactions: PriceEvidence | None
    current_listings: PriceEvidence | None
    confidence: str
    demo: bool = False
    warning: str | None = None

    def __post_init__(self) -> None:
        if self.actual_transactions is None and self.current_listings is None:
            raise ValueError("Market snapshot requires at least one evidence set")
        if self.confidence.casefold() not in {"low", "medium", "high"}:
            raise ValueError("Market confidence must be low, medium, or high")

    @classmethod
    def from_dict(
        cls,
        query: ComparableProperty,
        data: dict[str, Any],
    ) -> MarketSnapshot:
        if not isinstance(data, dict):
            raise MarketDataError("Market data response must be an object")
        actual = data.get("actual_transactions")
        listings = data.get("current_listings")
        try:
            return cls(
                query=query,
                actual_transactions=(
                    PriceEvidence.from_dict(actual)
                    if isinstance(actual, dict)
                    else None
                ),
                current_listings=(
                    PriceEvidence.from_dict(listings)
                    if isinstance(listings, dict)
                    else None
                ),
                confidence=str(data.get("confidence", "low")),
                demo=data.get("demo") is True,
                warning=(
                    str(data["warning"])
                    if data.get("warning") is not None
                    else None
                ),
            )
        except ValueError as error:
            raise MarketDataError("Market data response is invalid") from error

    def prompt_context(self) -> dict[str, Any]:
        return {
            "property": {
                "project": self.query.project,
                "cluster": self.query.cluster,
                "bedrooms": self.query.bedrooms,
                "property_type": self.query.property_type,
                "period_months": self.query.months,
            },
            "recent_actual_transactions": _evidence_dict(
                self.actual_transactions
            ),
            "current_asking_listings": _evidence_dict(self.current_listings),
            "confidence": self.confidence.casefold(),
            "demo": self.demo,
            "warning": self.warning,
            "required_language": (
                "Actual transactions are completed registered sales. Current listings "
                "are asking prices and must not be described as transactions."
            ),
        }

    def spoken_feedback(self) -> str:
        parts: list[str] = []
        if self.demo:
            parts.append(
                self.warning
                or "This is fictional demo data and not real market evidence."
            )
        if self.actual_transactions is not None:
            evidence = self.actual_transactions
            median = (
                f", with a median of {_format_aed(evidence.median_aed)}"
                if evidence.median_aed is not None
                else ""
            )
            parts.append(
                f"The recent actual registered transactions from {evidence.source} "
                f"show {evidence.count} comparable sales between "
                f"{_format_aed(evidence.low_aed)} and "
                f"{_format_aed(evidence.high_aed)}{median}."
            )
        if self.current_listings is not None:
            evidence = self.current_listings
            parts.append(
                f"Separately, {evidence.count} current asking listings from "
                f"{evidence.source} range from {_format_aed(evidence.low_aed)} to "
                f"{_format_aed(evidence.high_aed)}. Those are asking prices, not "
                "completed sales."
            )
        parts.append(
            "The exact unit location, plot, row, view, payment status, and handover "
            "position can change the comparison."
        )
        return " ".join(parts)


class MarketDataProvider(Protocol):
    async def get_comparables(
        self,
        query: ComparableProperty,
    ) -> MarketSnapshot | None: ...


class HttpMarketDataProvider:
    """Reads a normalized feed backed by approved transaction/listing providers."""

    def __init__(
        self,
        endpoint: str,
        *,
        bearer_token: str | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Market data endpoint must be an HTTP(S) URL")
        if parsed.scheme != "https" and parsed.hostname not in {
            "127.0.0.1",
            "localhost",
        }:
            raise ValueError("Remote market data endpoints must use HTTPS")
        if timeout_seconds <= 0:
            raise ValueError("Market data timeout must be positive")
        self.endpoint = endpoint
        self.bearer_token = bearer_token
        self.timeout_seconds = timeout_seconds

    async def get_comparables(
        self,
        query: ComparableProperty,
    ) -> MarketSnapshot | None:
        payload = await asyncio.to_thread(self._request, query)
        if payload is None:
            return None
        return MarketSnapshot.from_dict(query, payload)

    def _request(self, query: ComparableProperty) -> dict[str, Any] | None:
        parameters = urlencode(
            {
                "project": query.project,
                "cluster": query.cluster,
                "bedrooms": query.bedrooms,
                "property_type": query.property_type,
                "months": query.months,
            }
        )
        separator = "&" if "?" in self.endpoint else "?"
        headers = {"Accept": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        request = Request(
            f"{self.endpoint}{separator}{parameters}",
            headers=headers,
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
        except HTTPError as error:
            if error.code == 404:
                return None
            raise MarketDataError(
                f"Market data request failed with HTTP {error.code}"
            ) from error
        except (TimeoutError, URLError, OSError) as error:
            raise MarketDataError("Market data request failed") from error
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MarketDataError("Market data response is not valid JSON") from error
        if not isinstance(payload, dict):
            raise MarketDataError("Market data response must be an object")
        return payload


def _evidence_dict(evidence: PriceEvidence | None) -> dict[str, Any] | None:
    if evidence is None:
        return None
    return {
        "source": evidence.source,
        "as_of": evidence.as_of,
        "count": evidence.count,
        "low_aed": evidence.low_aed,
        "high_aed": evidence.high_aed,
        "median_aed": evidence.median_aed,
    }


def _format_aed(value: int | None) -> str:
    if value is None:
        raise ValueError("AED value is required")
    if value >= 1_000_000:
        amount = f"{value / 1_000_000:.2f}".rstrip("0").rstrip(".")
        return f"AED {amount} million"
    return f"AED {value:,}"