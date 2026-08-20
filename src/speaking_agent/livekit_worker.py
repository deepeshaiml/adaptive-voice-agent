from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time
from uuid import uuid4

from livekit.agents import AgentServer, AutoSubscribe, JobContext, cli

from speaking_agent.audio_recording import RecordingConsent, WaveConversationRecorder
from speaking_agent.adapters.asr.qwen_mlx import QwenMlxSpeechRecognizer
from speaking_agent.adapters.llm.qwen_mlx import MlxLmBackend, QwenMlxConversationModel
from speaking_agent.adapters.storage.sqlite import SQLiteCallRepository
from speaking_agent.adapters.telephony.livekit_room import LiveKitRoomTransport
from speaking_agent.adapters.tts.qwen_mlx import QwenMlxSpeechSynthesizer
from speaking_agent.answering import HeuristicAnsweringMachineDetector
from speaking_agent.campaign import load_campaign
from speaking_agent.delivery import campaign_voice_style
from speaking_agent.domain import ConversationContext
from speaking_agent.lead_workflow import (
    LeadDeliveryError,
    LeadWorkflowEvent,
    SalesCallAnalysis,
    WebhookLeadWorkflowSink,
)
from speaking_agent.market_data import HttpMarketDataProvider
from speaking_agent.outbound import (
    DialStatus,
    LiveKitSipDialer,
    OutboundDialRequest,
    allowed_test_numbers,
    ensure_controlled_test_number,
    ensure_permitted_call_time,
    final_outbound_result,
    mask_phone_number,
)
from speaking_agent.recording import completed_call_record, failed_call_record
from speaking_agent.records import SuppressionKeyMismatchError
from speaking_agent.suppression import (
    ContactAttemptLimitError,
    ContactAttemptPolicy,
    ContactSuppressedError,
    ContactSuppressionService,
)
from speaking_agent.voice_session import CallSession


logger = logging.getLogger("speaking-agent")
server = AgentServer()


@server.rtc_session(agent_name="speaking-agent")
async def entrypoint(context: JobContext) -> None:
    campaign_path = Path(
        os.environ.get(
            "SPEAKING_AGENT_CAMPAIGN",
            "campaigns/neoai_property_owner.json",
        )
    )

    metadata = json.loads(context.job.metadata or "{}")
    if not isinstance(metadata, dict):
        raise ValueError("LiveKit job metadata must be an object")
    phone_number = metadata.get("phone_number")
    raw_conversation_context = metadata.get("conversation_context", {})
    if not isinstance(raw_conversation_context, dict) or set(
        raw_conversation_context
    ) - {"recipient_name", "property_reference", "known_fields"}:
        raise ValueError("LiveKit conversation context is invalid")
    known_fields = raw_conversation_context.get("known_fields", {})
    if not isinstance(known_fields, dict):
        raise ValueError("LiveKit conversation context known_fields must be an object")
    conversation_context = ConversationContext(
        recipient_name=raw_conversation_context.get("recipient_name"),
        property_reference=raw_conversation_context.get("property_reference"),
        known_fields=known_fields,
    )
    participant_identity = (
        f"controlled-callee-{uuid4().hex[:12]}" if phone_number else None
    )
    dial_result = None
    attempt_policy = None
    suppression = None
    trunk_id = os.environ.get("SIP_OUTBOUND_TRUNK_ID", "")

    async def connect_room() -> None:
        nonlocal dial_result
        await context.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        if phone_number is not None:
            if attempt_policy is None:
                raise RuntimeError("Outbound attempt policy is not initialized")
            await attempt_policy.reserve(
                phone_number,
                session.conversation.state.call_id,
            )
            dialer = LiveKitSipDialer(context.api.sip)
            dial_result = await dialer.dial(
                OutboundDialRequest(
                    phone_number=phone_number,
                    room_name=context.room.name,
                    trunk_id=trunk_id,
                    participant_identity=participant_identity,
                    participant_attributes={"campaign.id": campaign_path.stem},
                )
            )
            if dial_result.status != DialStatus.CONNECTED:
                raise RuntimeError(
                    f"Outbound dial ended with {dial_result.status.value}"
                )

    async def wait_for_participant():
        return await context.wait_for_participant(identity=participant_identity)

    async def hang_up() -> None:
        await context.delete_room()

    transfer_to = os.environ.get("SPEAKING_AGENT_TRANSFER_TO")

    async def transfer() -> None:
        if participant_identity is None or not transfer_to:
            raise RuntimeError("Human transfer is not configured")
        destination = (
            transfer_to
            if transfer_to.startswith(("tel:", "sip:"))
            else f"tel:{transfer_to}"
        )
        await context.transfer_sip_participant(
            participant_identity,
            destination,
            play_dialtone=False,
        )

    async def persist_do_not_contact(call_id: str) -> None:
        if phone_number is None or suppression is None:
            return
        await suppression.suppress(phone_number, call_id)

    transport = LiveKitRoomTransport(
        room=context.room,
        connect_room=connect_room,
        wait_for_participant=wait_for_participant,
        hang_up_handler=hang_up,
        transfer_handler=transfer if transfer_to else None,
    )
    campaign = load_campaign(campaign_path)
    market_data_endpoint = os.environ.get("SPEAKING_AGENT_MARKET_DATA_URL")
    market_data_provider = (
        HttpMarketDataProvider(
            market_data_endpoint,
            bearer_token=os.environ.get("SPEAKING_AGENT_MARKET_DATA_TOKEN"),
            timeout_seconds=float(
                os.environ.get("SPEAKING_AGENT_MARKET_DATA_TIMEOUT_SECONDS", "5")
            ),
        )
        if market_data_endpoint
        else None
    )
    lead_workflow_endpoint = os.environ.get("SPEAKING_AGENT_LEAD_WORKFLOW_URL")
    lead_workflow_sink = (
        WebhookLeadWorkflowSink(
            lead_workflow_endpoint,
            bearer_token=os.environ.get("SPEAKING_AGENT_LEAD_WORKFLOW_TOKEN"),
            timeout_seconds=float(
                os.environ.get("SPEAKING_AGENT_LEAD_WORKFLOW_TIMEOUT_SECONDS", "5")
            ),
        )
        if lead_workflow_endpoint
        else None
    )
    audio_recorder = None
    recording_error: PermissionError | None = None
    if campaign.behavior["recording_enabled"]:
        consent_reference = metadata.get("recording_consent_reference")
        if not isinstance(consent_reference, str):
            recording_error = PermissionError(
                "Recording requires a per-call consent reference"
            )
        else:
            try:
                consent = RecordingConsent(consent_reference)
            except ValueError as error:
                recording_error = PermissionError(
                    "Recording consent reference is invalid"
                )
                recording_error.__cause__ = error
            else:
                audio_recorder = WaveConversationRecorder(
                    os.environ.get(
                        "SPEAKING_AGENT_RECORDING_DIRECTORY",
                        "data/recordings",
                    ),
                    consent,
                )
    session = CallSession(
        campaign=campaign,
        model=QwenMlxConversationModel(MlxLmBackend()),
        recognizer=QwenMlxSpeechRecognizer(),
        synthesizer=QwenMlxSpeechSynthesizer(
            default_style=campaign_voice_style(campaign)
        ),
        transport=transport,
        answering_detector=(
            HeuristicAnsweringMachineDetector() if phone_number else None
        ),
        on_do_not_contact=(persist_do_not_contact if phone_number else None),
        recognition_language="English",
        recognition_context=campaign.speech_recognition_context,
        transfer_available=bool(transfer_to),
        conversation_context=conversation_context,
        audio_recorder=audio_recorder,
        market_data_provider=market_data_provider,
    )
    legacy_retention = os.environ.get("SPEAKING_AGENT_LEGACY_RETENTION_DAYS")
    repository = SQLiteCallRepository(
        os.environ.get("SPEAKING_AGENT_DATABASE", "data/speaking_agent.db"),
        retention_days=campaign.behavior["data_retention_days"],
        legacy_retention_days=(
            int(legacy_retention) if legacy_retention else None
        ),
    )
    await repository.prepare()
    await repository.purge_expired()
    if not campaign.behavior["campaign_enabled"]:
        session.conversation.abort()
        try:
            await repository.save(
                failed_call_record(
                    session,
                    PermissionError("Campaign is disabled"),
                    connection_result="BLOCKED_CAMPAIGN_DISABLED",
                    duration_seconds=0.0,
                    phone_number_masked=(
                        mask_phone_number(phone_number) if phone_number else None
                    ),
                )
            )
        finally:
            await repository.close()
            context.shutdown("campaign disabled")
        return
    if recording_error is not None:
        session.conversation.abort()
        try:
            await repository.save(
                failed_call_record(
                    session,
                    recording_error,
                    connection_result="BLOCKED_RECORDING_CONSENT_REQUIRED",
                    duration_seconds=0.0,
                    phone_number_masked=(
                        mask_phone_number(phone_number) if phone_number else None
                    ),
                )
            )
        finally:
            await repository.close()
            context.shutdown("recording consent required")
        return
    if phone_number is not None:
        suppression = ContactSuppressionService(
            repository,
            os.environ.get("SPEAKING_AGENT_SUPPRESSION_KEY", ""),
        )
        attempt_policy = ContactAttemptPolicy(
            repository,
            os.environ.get("SPEAKING_AGENT_SUPPRESSION_KEY", ""),
            maximum_attempts=campaign.behavior["maximum_call_attempts"],
            window_hours=campaign.behavior["call_attempt_window_hours"],
            minimum_interval_minutes=campaign.behavior[
                "minimum_call_interval_minutes"
            ],
        )
        try:
            ensure_controlled_test_number(
                phone_number,
                allowed_test_numbers(
                    os.environ.get("SPEAKING_AGENT_ALLOWED_TEST_NUMBERS")
                ),
            )
            ensure_permitted_call_time(campaign.behavior)
            if not trunk_id.startswith("ST_"):
                raise ValueError("SIP_OUTBOUND_TRUNK_ID is not configured")
            await suppression.ensure_allowed(phone_number)
        except (
            ContactAttemptLimitError,
            ContactSuppressedError,
            PermissionError,
            SuppressionKeyMismatchError,
            ValueError,
        ) as error:
            is_suppressed = isinstance(error, ContactSuppressedError)
            is_attempt_block = isinstance(error, ContactAttemptLimitError)
            is_policy_block = isinstance(error, PermissionError) and not (
                is_suppressed or is_attempt_block
            )
            blocked_outcome = "DO_NOT_CONTACT" if is_suppressed else "UNKNOWN"
            session.conversation.abort(blocked_outcome)
            try:
                await repository.save(
                    failed_call_record(
                        session,
                        error,
                        connection_result=(
                            "BLOCKED_DO_NOT_CONTACT"
                            if is_suppressed
                            else (
                                "BLOCKED_ATTEMPT_POLICY"
                                if is_attempt_block
                                else (
                                    "BLOCKED_POLICY"
                                    if is_policy_block
                                    else "BLOCKED_CONFIGURATION"
                                )
                            )
                        ),
                        duration_seconds=0.0,
                        phone_number_masked=mask_phone_number(phone_number),
                    )
                )
            finally:
                await repository.close()
                context.shutdown(
                    "contact suppressed"
                    if is_suppressed
                    else "outbound policy blocked"
                )
            return
    started_at = time.perf_counter()
    masked_number = mask_phone_number(phone_number) if phone_number else None
    try:
        result = await session.run()
    except BaseException as error:
        if isinstance(error, ContactSuppressedError):
            session.conversation.abort("DO_NOT_CONTACT")
            connection_result = "BLOCKED_DO_NOT_CONTACT"
        elif isinstance(error, ContactAttemptLimitError):
            connection_result = "BLOCKED_ATTEMPT_POLICY"
        elif dial_result is not None and dial_result.status != DialStatus.CONNECTED:
            connection_result = final_outbound_result(dial_result.status).value
        else:
            connection_result = "FAILED"
        if dial_result is not None and dial_result.status == DialStatus.CONNECTED:
            try:
                await context.delete_room()
            except Exception:
                pass
        if (
            suppression is not None
            and session.conversation.state.do_not_contact
        ):
            await suppression.suppress(
                phone_number,
                session.conversation.state.call_id,
            )
        await repository.save(
            failed_call_record(
                session,
                error,
                connection_result=connection_result,
                duration_seconds=time.perf_counter() - started_at,
                phone_number_masked=masked_number,
            )
        )
        raise
    else:
        connection_result = (
            final_outbound_result(
                dial_result.status,
                result.answer_kind,
                disconnected=result.disconnected,
            ).value
            if dial_result is not None
            else "ANSWERED_HUMAN"
        )
        if suppression is not None and result.lead.outcome == "DO_NOT_CONTACT":
            await suppression.suppress(
                phone_number,
                session.conversation.state.call_id,
            )
        record = completed_call_record(
            session,
            result,
            connection_result=connection_result,
            duration_seconds=time.perf_counter() - started_at,
            phone_number_masked=masked_number,
        )
        await repository.save(record)
        if lead_workflow_sink is not None and result.answer_kind.value == "HUMAN":
            try:
                await lead_workflow_sink.publish(
                    LeadWorkflowEvent(
                        call_id=record.call_id,
                        campaign_id=record.campaign_id,
                        owner_name=conversation_context.recipient_name,
                        phone_number=phone_number,
                        phone_number_masked=record.phone_number_masked,
                        analysis=SalesCallAnalysis.from_dict(
                            record.sales_summary,
                            summary_text=record.summary,
                        ),
                        transcript=record.transcript,
                        recording_url=record.recording_url,
                    )
                )
            except LeadDeliveryError:
                logger.exception(
                    "lead workflow delivery failed for call %s",
                    record.call_id,
                )
    finally:
        try:
            await repository.close()
        finally:
            context.shutdown("call ended")

    logger.info(
        "call completed %s",
        json.dumps(
            {
                "call_id": session.conversation.state.call_id,
                "campaign_id": session.conversation.state.campaign_id,
                "session_id": session.conversation.state.session_id,
                "outcome": result.lead.outcome,
                "answer_kind": result.answer_kind,
                "outbound_result": (
                    connection_result
                    if dial_result is not None
                    else None
                ),
                "callee": masked_number,
                "interruptions": result.interruptions,
                "disconnected": result.disconnected,
                "cleanup_errors": result.cleanup_errors,
            }
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
