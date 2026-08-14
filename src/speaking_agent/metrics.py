from __future__ import annotations

from dataclasses import dataclass, field

from speaking_agent.records import CallRecord


_ANSWERED_RESULTS = {
    "ANSWERED_HUMAN",
    "VOICEMAIL",
    "ANSWERED_IVR",
    "MACHINE_UNAVAILABLE",
    "ANSWERED_UNCERTAIN",
    "DROPPED",
}


@dataclass(frozen=True, slots=True)
class CallMetrics:
    records: int
    calls_attempted: int
    answered: int
    answered_human: int
    voicemail: int
    qualified: int
    sell: int
    rent: int
    callbacks: int
    not_interested: int
    do_not_contact: int
    failures: int
    average_duration_seconds: float
    average_latencies_seconds: dict[str, float] = field(default_factory=dict)


def aggregate_call_metrics(records: list[CallRecord]) -> CallMetrics:
    attempted = [
        record
        for record in records
        if not record.connection_result.startswith("BLOCKED_")
    ]
    latency_values: dict[str, list[float]] = {}
    for record in attempted:
        for name, value in record.latencies.items():
            latency_values.setdefault(name, []).append(value)

    return CallMetrics(
        records=len(records),
        calls_attempted=len(attempted),
        answered=sum(
            record.connection_result in _ANSWERED_RESULTS for record in attempted
        ),
        answered_human=sum(
            record.connection_result == "ANSWERED_HUMAN" for record in attempted
        ),
        voicemail=sum(record.connection_result == "VOICEMAIL" for record in attempted),
        qualified=sum(record.qualified for record in attempted),
        sell=sum(record.outcome == "SELL" for record in attempted),
        rent=sum(record.outcome == "RENT" for record in attempted),
        callbacks=sum(record.callback_requested for record in attempted),
        not_interested=sum(record.outcome == "NOT_INTERESTED" for record in attempted),
        do_not_contact=sum(record.outcome == "DO_NOT_CONTACT" for record in records),
        failures=sum(
            record.connection_result not in _ANSWERED_RESULTS
            and not record.connection_result.startswith("BLOCKED_")
            for record in attempted
        ),
        average_duration_seconds=(
            sum(record.duration_seconds for record in attempted) / len(attempted)
            if attempted
            else 0.0
        ),
        average_latencies_seconds={
            name: sum(values) / len(values)
            for name, values in sorted(latency_values.items())
        },
    )
