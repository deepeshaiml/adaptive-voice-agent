from __future__ import annotations

from speaking_agent.observability import LatencyTrace, TimingEventName
from speaking_agent.records import CallRecord
from speaking_agent.voice_session import CallSession, VoiceCallResult


def latency_snapshot(trace: LatencyTrace) -> dict[str, float]:
    measurements = {
        "speech_end_to_asr_partial": trace.latest_duration(
            TimingEventName.SPEECH_END,
            TimingEventName.ASR_PARTIAL,
        ),
        "speech_end_to_asr_final": trace.latest_duration(
            TimingEventName.SPEECH_END,
            TimingEventName.ASR_FINAL,
        ),
        "llm_first_token": trace.latest_duration(
            TimingEventName.LLM_START,
            TimingEventName.LLM_FIRST_TOKEN,
        ),
        "tts_first_audio": trace.latest_duration(
            TimingEventName.TTS_START,
            TimingEventName.TTS_FIRST_AUDIO,
        ),
        "speech_end_to_playback": trace.latest_duration(
            TimingEventName.SPEECH_END,
            TimingEventName.PLAYBACK_START,
        ),
    }
    return {
        name: duration
        for name, duration in measurements.items()
        if duration is not None
    }


def completed_call_record(
    session: CallSession,
    result: VoiceCallResult,
    *,
    connection_result: str,
    duration_seconds: float,
    phone_number_masked: str | None,
) -> CallRecord:
    state = session.conversation.state
    return CallRecord(
        call_id=state.call_id,
        session_id=state.session_id,
        campaign_id=state.campaign_id,
        connection_result=connection_result,
        outcome=result.lead.outcome,
        qualified=result.lead.qualified,
        summary=result.lead.summary,
        fields=result.lead.fields,
        callback_requested=result.lead.callback_requested,
        human_followup_required=result.lead.human_followup_required,
        answer_kind=result.answer_kind.value,
        phone_number_masked=phone_number_masked,
        interruptions=result.interruptions,
        disconnected=result.disconnected,
        duration_seconds=duration_seconds,
        latencies=latency_snapshot(session.trace),
        error=(";".join(result.cleanup_errors) or None),
    )


def failed_call_record(
    session: CallSession,
    error: BaseException,
    *,
    connection_result: str,
    duration_seconds: float,
    phone_number_masked: str | None,
) -> CallRecord:
    state = session.conversation.state
    qualified = state.outcome in session.conversation.campaign.qualified_outcomes
    summary = f"Call failed after outcome {state.outcome}."
    return CallRecord(
        call_id=state.call_id,
        session_id=state.session_id,
        campaign_id=state.campaign_id,
        connection_result=connection_result,
        outcome=state.outcome,
        qualified=qualified,
        summary=summary,
        fields=dict(state.fields),
        callback_requested=state.callback_requested,
        human_followup_required=(
            state.outcome in session.conversation.campaign.human_followup_outcomes
        ),
        answer_kind=(session._answer_kind.value if session._answer_kind else None),
        phone_number_masked=phone_number_masked,
        interruptions=session.interruptions,
        duration_seconds=duration_seconds,
        latencies=latency_snapshot(session.trace),
        error=type(error).__name__,
    )
