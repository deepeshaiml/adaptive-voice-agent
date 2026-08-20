# Adaptive Voice Agent

A local-first, provider-neutral framework for adaptive voice conversations. It includes
a campaign-driven conversation core, local Qwen LLM/ASR/TTS adapters, microphone and
speaker interaction, a real-time call lifecycle, LiveKit room transport, outbound SIP,
structured SQLite outcomes, operator tooling, and compliance-focused safeguards.

The distribution and repository use the general name `adaptive-voice-agent`. The
internal Python namespace remains `speaking_agent` for compatibility.

The complete local inference stack is currently verified on Apple-silicon macOS. See
[ROADMAP.md](ROADMAP.md) for the ordered cross-platform, CUDA, CPU, audio, packaging,
benchmarking, and deployment TODOs.

A real PSTN call has not been placed from this repository because that requires private
LiveKit credentials, an outbound trunk, and a number explicitly controlled by the
operator. The command is implemented but deliberately blocked without all three.

## Current Status

Status last verified on **2026-08-20** on Apple-silicon macOS with Python 3.12:

| Area | Status |
|---|---|
| Campaign-driven conversation core | Complete and covered by scenario tests |
| Deterministic safety and field grounding | Complete for configured policies |
| Local Qwen LLM, ASR, and TTS | Verified with the pinned MLX models |
| Local microphone/speaker conversation | Verified in half-duplex and speaker modes |
| Local full duplex and barge-in | Implemented and tested; echo suppression remains experimental |
| LiveKit room audio | Verified bidirectionally |
| Outbound SIP/PSTN | Implemented behind controlled-call gates; real PSTN not yet exercised |
| SQLite outcomes, suppression, retention, and metrics | Complete and tested |
| DAMAC seller summaries, P1-P4 follow-ups, and lead webhook | Implemented and unit tested; external CRM not exercised |
| Approved market-data gateway | Implemented and unit tested; provider credentials/feed not supplied |
| Linux, Windows, CUDA, and portable inference | Not yet integrated or natively verified |

The current validation passes **256 tests**, `pip check`, bytecode compilation, diff
validation, local Markdown-link checks, and editor diagnostics. A real local Qwen
planning smoke completed in 1.214 seconds for one warm single-turn scenario; this is a
smoke measurement, not a P50/P95 benchmark. Independent blocker review found no release
blockers for the current local prototype scope.

This is not yet a production telephony release. Production readiness still requires a
controlled PSTN call, native platform evidence, measured latency and memory budgets,
production acoustic echo cancellation/VAD, and authoritative legal/compliance review.

## Architecture

```text
LiveKit / SIP -------> CallTransport adapter --------+
Incoming PCM -------> Turn detection                 |
                                                     v
Campaign JSON ---> ConversationPolicy ---> CallSession
                         |                    |
                         v                    v
               ConversationState      ASR / LLM / TTS protocols
                         |                    ^
                         v                    |
                    LeadOutcome        mock / MLX adapters
                      |
                      +----> MarketDataProvider ---> approved API/feed gateway
                      |
                      v
                     CallRepository ---> SQLite ---> LeadWorkflowSink ---> CRM/Yasir
```

The domain, policy, conversation, and call lifecycle own application types. They do not
import LiveKit, SIP, MLX, Qwen, SQLite, cloud SDKs, or OS-specific APIs. Provider types
are translated only inside adapters.

## Setup

The package requires Python 3.11+ and is currently verified on Python 3.12. Install the
full verified local stack in the configured virtual environment:

```sh
.venv/bin/python -m pip install -e '.[realtime]'
```

Pinned integration versions are `mlx-lm==0.31.3`, `mlx-audio==0.4.8`,
`livekit-agents==1.6.10`, and its required `livekit==1.1.14`.

Run the complete external-service-free test suite:

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Run the standard command-line quality gate before merging a behavioral change:

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
.venv/bin/python -m compileall -q src scripts
git diff --check
```

For documentation changes, also verify local Markdown links and editor diagnostics with
the available IDE/tooling. For platform or adapter changes, the native checks listed in
the validation table below remain mandatory.

An editable install also provides the preferred public commands:

| Command | Purpose |
|---|---|
| `adaptive-voice-agent` | Text simulator with mock or MLX conversation model |
| `adaptive-voice-speech` | WAV-based ASR/TTS diagnostics |
| `adaptive-voice-local` | Local microphone/speaker conversation |
| `adaptive-voice-calls` | Inspect persisted outcomes and aggregate metrics |
| `adaptive-voice-retention` | Purge expired structured records |

The older `speaking-agent-*` command aliases and `python -m speaking_agent...` module
forms remain available for compatibility.

## Text Simulation

Run deterministically without models, speech, telephony, a database, or Internet:

```sh
PYTHONPATH=src .venv/bin/python -m speaking_agent
```

Use the local Qwen model instead:

```sh
PYTHONPATH=src .venv/bin/python -m speaking_agent --model mlx
```

The default checkpoint is `mlx-community/Qwen3-4B-Instruct-2507-4bit` (about
2.26 GB). The simulator uses the same `ConversationSession` and campaign policy as a
voice call. Type `/quit` to end an unfinished simulation.

## Local Speech

Generate PCM WAV audio with the macOS debug voice or Qwen3-TTS:

```sh
PYTHONPATH=src .venv/bin/python -m speaking_agent.speech_cli synthesize \
  --adapter system --text "Local speech test" --output output/system.wav

PYTHONPATH=src .venv/bin/python -m speaking_agent.speech_cli synthesize \
  --adapter qwen --voice Aiden --language English \
  --text "I might sell in two months" --output output/qwen.wav
```

Transcribe a 16-bit PCM WAV with Qwen3-ASR:

```sh
PYTHONPATH=src .venv/bin/python -m speaking_agent.speech_cli transcribe \
  --language English --input output/qwen.wav
```

The harness reports TTS first-audio/total time and ASR first-partial/final time. The
default checkpoints are `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit`
(about 1.97 GB) and `mlx-community/Qwen3-ASR-0.6B-8bit` (about 1.01 GB).

## Local Voice Conversation

Talk to the complete local Qwen stack through the Mac microphone and speakers:

```sh
PYTHONPATH=src .venv/bin/python -m speaking_agent.local_voice_chat
```

The agent speaks the campaign opening, listens until 550 ms of trailing silence,
prints what Qwen3-ASR heard, runs the real conversation model, and speaks the Qwen3-TTS
response. Each turn reports ASR, LLM, TTS-first-audio, and speech-end-to-response
latency. The local helper applies a warm conversational speaking style by default; use
`--style` to replace it. Audio and transcripts are not written to disk by default.
Press `Ctrl-C` to stop.

On first use, allow microphone access for VS Code or the terminal in macOS **System
Settings > Privacy & Security > Microphone**. Headphones prevent the microphone from
hearing the agent's speaker output. List and select CoreAudio devices with:

```sh
PYTHONPATH=src .venv/bin/python -m speaking_agent.local_voice_chat --list-devices
PYTHONPATH=src .venv/bin/python -m speaking_agent.local_voice_chat \
  --input-device 2 --output-device 3
```

For telephone-like conversation, use full-duplex mode:

```sh
.venv/bin/adaptive-voice-local --full-duplex
```

To practice the personalized three-stage opening with fake DAMAC data:

```sh
.venv/bin/adaptive-voice-local --full-duplex --demo-metadata
```

That demo identifies the disclosed AI assistant, asks for Mr. Ahmed, mentions a fake
4-bedroom townhouse in Nice only after recipient confirmation, asks whether it is a good
time, and then starts qualification. NeoAI is the caller; Yasir is the human who receives
the post-call lead. When the two loopback demo services below are running,
`--demo-metadata` automatically reads fictional market data from port `8765`, uses the
fixed fake number `+971500000000`, and sends Yasir's notification to port `8766` only
after the conversation reaches its closing outcome. The terminal then prints
`Yasir notification sent`. The configured campaign opening is used when metadata is
omitted.

Supply reviewed metadata explicitly when it becomes available:

```sh
.venv/bin/adaptive-voice-local --full-duplex \
  --recipient-name "Ms. Fatima" \
  --property-reference "your townhouse in Malta, DAMAC Lagoons" \
  --phone-number "+971501234567" \
  --project "DAMAC Lagoons" --cluster Malta --bedrooms 4 \
  --property-type townhouse
```

Recipient name and property reference must be supplied together. The reference gates the
personalized opening; the recipient name and verified structured fields may appear in the
retained call summary. Never populate metadata from an unverified enrichment source, and
apply the same access and deletion policy used for transcripts and call outcomes.
`--phone-number` must use E.164 format and is sent only to a configured lead workflow;
omit it when that workflow does not need a WhatsApp action.

### Consent-Gated Audio Recording

Record a local full-duplex quality session only after obtaining and documenting the
required consent:

```sh
.venv/bin/adaptive-voice-local --full-duplex --demo-metadata \
  --record-audio \
  --recording-consent-reference local-self-test-20260816
```

The consent reference is an external audit identifier, not proof that consent was
legally sufficient. The operator remains responsible for approved notice, consent,
access, training use, and deletion rules. Local recording requires full duplex; without
`--record-audio`, no audio file is created.

Artifacts are written beneath `data/recordings/YYYY-MM-DD/` by default:

- `<call-id>.wav`: 16 kHz stereo PCM, owner on channel 1 (left) and agent on channel 2
  (right). The agent channel accepts only transport-confirmed playout; queued audio
  discarded by barge-in is not recorded.
- `<call-id>.json`: consent reference, purpose, campaign/call IDs, timestamps, expiry,
  channel map, duration, and SHA-256 digest. It contains no transcript, phone number,
  recipient name, or property reference.

Directories are owner-only (`0700`) and files are owner-only (`0600`). Raw voice can
still identify a person and contain sensitive statements. The application does not
encrypt or de-identify recordings, label training examples, or establish rights to train
a model. Use encrypted storage and a separately reviewed de-identification/access/export
workflow before moving recordings off the controlled machine or using them for training.
Recording creation and retention refuse symlinked storage components, and an active-call
marker prevents retention from removing an in-progress call directory. Stale crash
partials/orphan WAVs are removed after the retention worker's grace period.

The microphone remains active while the agent speaks. Confirmed near-end speech stops
the current TTS playback, preserves the interruption, and immediately enters the normal
ASR/LLM/TTS flow. The terminal reports the transcript and interruption count without a
separate “Listening” phase.

The personalized recipient-confirmation opening is a deliberate exception: it must finish
before barge-in is enabled because it contains required disclosure and gates private
property metadata. Audio spoken over that opening is queued; meaningful replies such as
“Yes” are processed immediately after delivery, while low-information echo-like ASR
fragments such as “Ah”, “Um”, “Hi”, or “Hello” are ignored. Normal barge-in resumes for
all later turns. Unrecognized confirmation replies are bounded and cannot replay the
opening indefinitely.

Terminal hangup also checks for microphone speech that began just before closing
playout completed. A short bounded grace lets candidate speech cross the VAD threshold;
confirmed speech is then allowed to finish so late do-not-contact requests can override
the prior outcome and persist before hangup. Calls with no pending microphone activity
still hang up immediately after playout.

Privacy-sensitive replies are also deferred from the first candidate speech frame, not
only after VAD confirmation. This prevents an earlier queued “Yes” from causing property
metadata playback while a later wrong-recipient correction is still being spoken.

With speakers, full-duplex mode uses the outgoing PCM as an echo reference and rejects
correlated or low-energy loopback. This is an experimental software echo suppressor, not
production-grade acoustic echo cancellation. Keep speaker volume moderate. If the agent
interrupts itself, increase `--barge-in-energy-threshold` or `--echo-gain`; if your voice
cannot interrupt it, lower those values. `--echo-correlation-threshold` controls how
closely microphone audio must match recent speaker output before it is rejected.

Use half-duplex speaker mode as the stable fallback in a difficult room:

```sh
.venv/bin/adaptive-voice-local --speaker-mode
```

The microphone stays closed while the agent speaks, waits 500 ms for room echo to
settle, and then starts listening automatically. Keep speaker volume moderate. In a very
echoey room, increase the guard with `--speaker-settle-ms 1000`. macOS default devices
are preferred because CoreAudio numeric indices change when monitors, docks, or headsets
connect. Use `--list-devices` and explicit device names only when defaults are wrong.
The local VAD accepts speech as short as 60 ms so one-word replies such as “yes”, “no”,
or “both” are retained; raise `--minimum-speech-ms` if brief room noise triggers it.

If normal speech is not detected, lower `--energy-threshold` from `0.02` to `0.01`.
Raise it in a noisy room. Voice character can be tested with `--voice` and `--style`:

```sh
PYTHONPATH=src .venv/bin/python -m speaking_agent.local_voice_chat \
  --voice Aiden --style "Speak warmly, naturally, and concisely."
```

Qwen TTS now defaults to greedy decoding (`--tts-temperature 0`) so the same
text/voice/style produces repeatable PCM and is much less likely to drift in speaker
identity or delivery between turns. A native check on the pinned checkpoint produced
identical hashes for two repeated adapter generations. To deliberately restore more
prosodic variation, use for example `--tts-temperature 0.3 --tts-top-k 30`; sampled
decoding can also reintroduce audible voice/style variation.

Both modes use the same campaign, conversation policy, local models, and structured call
state. LiveKit is the intended telephony transport path, where endpoint/WebRTC echo
cancellation is expected to handle acoustic loopback.

## LiveKit Worker

Create `.env` from `.env.example` and provide your own values. At minimum, room tests
need `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET`.

Run the agent worker:

```sh
PYTHONPATH=src .venv/bin/python -m speaking_agent.livekit_worker dev
```

For a local transport integration test, start the server in another terminal and run
the bidirectional audio smoke script:

```sh
livekit-server --dev
PYTHONPATH=src .venv/bin/python scripts/livekit_audio_smoke.py
```

The worker receives 16 kHz mono PCM, detects turns, supports barge-in by cancelling TTS
and clearing the LiveKit audio queue, publishes 24 kHz mono PCM, waits for playout before
hangup/transfer, and releases all session resources on exit.

### Complete Local Demo

The repository includes a loopback-only fake stack for development without DLD, portal,
CRM, WhatsApp, LiveKit, or PSTN credentials. Use only fake owners and controlled test
numbers. Every market response begins with `FICTIONAL DEMO DATA`, uses `DEMO ONLY`
sources, and is rejected automatically when `controlled_test_mode` is false.

Start the fictional market service:

```sh
PYTHONPATH=src .venv/bin/python scripts/demo_market_data_server.py
```

It serves three fixtures: Nice 4BR townhouse, Malta 4BR townhouse, and Venice 6BR villa.
Unknown combinations return no comparables instead of fabricated fallback prices.

In a second terminal, start the local Yasir/CRM receiver:

```sh
PYTHONPATH=src .venv/bin/python scripts/demo_lead_workflow_server.py
```

It prints a compact Yasir notification, deduplicates by `call_id`, and appends the full
event to the owner-only ignored file `data/demo/lead_events.jsonl`. The latest received
event is available locally at `http://127.0.0.1:8766/latest`; the latest event that
actually requires a Yasir alert remains available at
`http://127.0.0.1:8766/latest-yasir`. Unqualified and not-interested calls are stored but
do not replace that alert view.

With both services running, start the real local Qwen voice conversation:

```sh
PYTHONPATH=src .venv/bin/python -m speaking_agent.local_voice_chat \
  --full-duplex --demo-metadata
```

Answer the recipient and timing confirmations, complete the seller questions, and let
the agent deliver its closing line. The latest Yasir event changes only when the call is
complete; stopping with `Ctrl-C` before a final outcome does not create a lead.
For the identity question, say `Yes` or `Yes, speaking`; `Okay` is only an acknowledgement
and intentionally does not unlock the private property-reference step.

Alternatively, run a complete synthetic hot-seller flow without microphone or models:

```sh
PYTHONPATH=src .venv/bin/python scripts/demo_sales_flow.py \
  --call-id demo-nice-hot-seller-1
```

Change `--call-id` to create another event. To connect the running LiveKit worker to the
same fake services for a controlled call, configure:

```text
SPEAKING_AGENT_MARKET_DATA_URL=http://127.0.0.1:8765/comparables
SPEAKING_AGENT_LEAD_WORKFLOW_URL=http://127.0.0.1:8766/events
```

The WhatsApp action is only a generated `wa.me` link and only appears after explicit
permission and number confirmation. The demo receiver never sends a WhatsApp message.
Do not expose either server beyond loopback or use its fictional evidence in a real owner
conversation.

### Approved Market Data

Set `SPEAKING_AGENT_MARKET_DATA_URL` to an internal HTTPS gateway backed by approved DLD,
Property Monitor, REIDIN, or brokerage listing feeds. The application does not scrape
Property Finder or Bayut. It sends `project`, `cluster`, `bedrooms`, `property_type`, and
`months` as query parameters after the comparable property is identified. An optional
bearer token comes from `SPEAKING_AGENT_MARKET_DATA_TOKEN`.

The gateway response is normalized so completed sales cannot be confused with asking
prices:

```json
{
  "actual_transactions": {
    "source": "Dubai Land Department",
    "as_of": "2026-08-20",
    "count": 9,
    "low_aed": 3550000,
    "high_aed": 3950000,
    "median_aed": 3750000
  },
  "current_listings": {
    "source": "Approved brokerage feed",
    "as_of": "2026-08-20",
    "count": 12,
    "low_aed": 3850000,
    "high_aed": 4200000,
    "median_aed": 4000000
  },
  "confidence": "high"
}
```

Either evidence section may be omitted, but source and `as_of` are mandatory when it is
present. Remote endpoints must use HTTPS. Missing or failed market data never becomes a
model-generated estimate; qualification continues without the unavailable evidence.

### CRM And Yasir Notification

Set `SPEAKING_AGENT_LEAD_WORKFLOW_URL` to an authorized HTTPS webhook. After each
connected human call, the worker first stores the call record and then posts the full
structured summary, transcript when enabled, recording URL when available, P1-P4
priority, follow-up timestamp/task flag, recommended action, and notification mode.

P1 hot sellers and P2 owners open at the right price set `notify_yasir=true` with
`notification_mode=IMMEDIATE`. P3 potential sellers use `MARKET_FOLLOW_UP`. P4 future
sellers create a dated follow-up task without an urgent notification. Qualified events
include an `open_whatsapp_url` only after the owner grants permission and confirms the
called number. The raw number is sent only to the
configured workflow endpoint and is not persisted in SQLite; stored records retain the
masked number. A failed webhook is logged and does not delete or roll back the call
record. Production deployments should place the endpoint behind a durable CRM/outbox
with authentication, retry, idempotency by `call_id`, and role-based access.

## Controlled Calls

Outbound calling is restricted to controlled tests. Configure:

```text
SIP_OUTBOUND_TRUNK_ID=ST_...
SPEAKING_AGENT_ALLOWED_TEST_NUMBERS=+<controlled-E.164-number>
SPEAKING_AGENT_SUPPRESSION_KEY=<at-least-32-random-bytes>
```

Keep the worker running, then inspect a masked dry run:

```sh
PYTHONPATH=src .venv/bin/python -m speaking_agent.call_cli +<controlled-number>
```

Only after verifying the room, trunk, campaign identity, number, and calling policy:

```sh
PYTHONPATH=src .venv/bin/python -m speaking_agent.call_cli \
  +<controlled-number> --execute
```

Optional reviewed personalization uses the same flags as local mode and is forwarded in
private LiveKit job metadata; the dry-run summary reports only whether personalization is
enabled:

```sh
PYTHONPATH=src .venv/bin/python -m speaking_agent.call_cli \
  +<controlled-number> \
  --recipient-name "Mr. Ahmed" \
  --property-reference "your townhouse in Nice, DAMAC Lagoons" \
  --project "DAMAC Lagoons" --cluster Nice --bedrooms 4 \
  --property-type townhouse
```

For a controlled LiveKit/SIP recording, first set `behavior.recording_enabled` to `true`
in the reviewed campaign, configure `SPEAKING_AGENT_RECORDING_DIRECTORY`, and dispatch
with a per-call consent reference:

```sh
PYTHONPATH=src .venv/bin/python -m speaking_agent.call_cli \
  +<controlled-number> \
  --recording-consent-reference consent-ticket-42 \
  --execute
```

The worker blocks before dialing when recording is enabled but the consent reference is
missing or invalid. A reference is required even if the campaign opening also contains a
recording notice; configure both according to authoritative requirements.

The worker blocks non-allowlisted numbers, invalid E.164 input, active do-not-contact
records, excessive attempts, too-short retry intervals, missing suppression keys, and
unconfigured trunks. Non-test campaigns must additionally configure a reviewed timezone
and local calling window. No jurisdiction-specific hours are invented by this example.
The suppression key is bound to the SQLite database by a non-secret identifier; changing
it fails closed. Key rotation requires an explicit migration, not a silent environment
variable replacement.

Outbound sessions listen before speaking to separate a human from explicit voicemail or
IVR signatures. Human speech then receives the configured identity disclosure before the
adaptive response. Voicemail uses a separate campaign message. Optional cold transfer
uses `SPEAKING_AGENT_TRANSFER_TO` and requires provider support for SIP REFER.
When transfer is not configured, the agent uses the campaign's transfer-unavailable
message and hangs up safely instead of announcing a transfer it cannot perform.

## Outcomes

Inspect recent structured call records:

```sh
PYTHONPATH=src .venv/bin/python -m speaking_agent.operator_cli list
PYTHONPATH=src .venv/bin/python -m speaking_agent.operator_cli show <call-id>
PYTHONPATH=src .venv/bin/python -m speaking_agent.operator_cli metrics
```

SQLite defaults to `data/speaking_agent.db`. Records include the mandatory sectioned
seller summary, P1-P4 priority, follow-up timestamp, validated property/price/WhatsApp
fields, market evidence discussed, connection data, duration, and latency metrics.
Campaigns may opt into full transcript persistence with `behavior.transcript_enabled`;
the DAMAC campaign enables it and includes a transcription notice. Transcripts and local
recording references use the same protected row and campaign retention horizon. Raw
telephone numbers are never persisted. Do-not-contact and attempt tracking use a keyed
HMAC fingerprint. Structured call records and attempt history are assigned per-row
expirations using their campaign's `data_retention_days`; recording manifests use the
same campaign horizon, while suppression entries are retained. Run the
retention service beside the call worker so expired structured rows, recording WAVs, and
manifests are removed even while no calls are arriving:

```sh
PYTHONPATH=src .venv/bin/python -m speaking_agent.retention_worker \
  --database data/speaking_agent.db \
  --recording-directory data/recordings \
  --interval-seconds 3600
```

Populated databases created before per-row expiration fail closed. Migrate once with a
reviewed horizon at least as long as every legacy campaign policy, then remove the option:

```sh
PYTHONPATH=src .venv/bin/python -m speaking_agent.retention_worker \
  --database data/speaking_agent.db --legacy-retention-days 90 --once
```

An opt-out is written through the suppression repository immediately upon recognition,
before its spoken acknowledgement and call teardown.

## Campaigns

Campaigns are runtime JSON files. Question order and wording come from configuration
rather than a hardcoded sequence. The current deterministic grounding policy still
contains property-domain evidence rules for fields such as intent, location, property
type, price, timeline, and listing state. Add a pluggable or declarative domain policy
before treating JSON alone as sufficient for a non-property campaign.

The default simulator, local voice, and LiveKit campaign is
`campaigns/neoai_property_owner.json`. That is where you
give the agent its overall goal and flexible scenario guidance:

The repository includes two examples:

- `campaigns/neoai_property_owner.json`: DAMAC Lagoons seller qualification, market
  feedback, WhatsApp/document permission, and Yasir follow-up.
- `campaigns/property_owner.json`: legacy generic sell/rent property campaign.

```json
{
  "objective": "The measurable business goal and acceptable outcomes.",
  "conversation_brief": "Who the agent is, who it is speaking with, useful context, and what success means.",
  "conversation_guidelines": [
    "How to sound and behave across every turn.",
    "Answer what the person said before returning gently to the goal."
  ],
  "scenario_playbook": [
    {
      "when": "A situation the person may introduce.",
      "strategy": "What the agent should achieve or avoid, without prescribing exact words."
    }
  ]
}
```

`objective`, outcomes, fields, hard stops, and prohibited statements are firm policy.
`conversation_brief`, `conversation_guidelines`, and `scenario_playbook` guide Qwen's
judgment. They are explicitly passed as strategies rather than dialogue to quote, so the
agent can answer unexpected turns and vary its language while staying on goal. Exact
approved facts belong in `faq_answers`; standalone ASR/paraphrase routes to those facts
belong in `faq_aliases`; alternate follow-up wording belongs in `question_variants`.
FAQ aliases are exact punctuation-insensitive routes, so mixed FAQ-and-intent turns still
go through the model with the approved answer in context. Put a canonical FAQ question
in `faq_answer_only` when its approved answer should be spoken without immediately
appending the next qualification question, such as identity clarification or a complaint
about repetition. Compliant `opening_variants` are selected per session so repeated calls
do not always begin with identical wording.

The DAMAC campaign pursues selling intention, exact unit details, asking/minimum price,
grounded market feedback, WhatsApp permission, documents, and Yasir follow-up in that
order without treating the questions as a literal script. `field_dependencies` suppress
WhatsApp number and document questions when permission is declined. Market and outbound
workflow capabilities exist only when their approved endpoints are configured. An
explicit refusal ends the call without further probing, and an explicit future-contact
refusal additionally enters do-not-contact suppression.

During a session, the model receives bounded in-memory dialogue history for both the
owner and agent, plus the current stage, prior question counts, skipped fields, and known
structured facts. This lets it resolve references, remember what both sides said, and
avoid repeating an earlier answer or question. Configure the owner-turn bound with
`behavior.conversation_memory_turns` (default `12`). This prompt window remains bounded.
The independent full dialogue is written to the retained call record only when
`behavior.transcript_enabled` is true.

To create another campaign:

1. Copy `campaigns/property_owner.json` and assign a unique `campaign_id`.
2. Configure the objective, introduction, full opening, and exact disclosure fragments.
3. Define outcomes, outcome guidance, branch fields, field types, and one question per
   field.
4. Configure hard stops, FAQs, prohibited statements, terminal/closing behavior,
   voicemail text, attempt limits, retention, and model failure bounds.
    Set `transcript_enabled` explicitly and include the reviewed notice/consent language
    required for the deployment. Set `recording_enabled` explicitly. `true` enables
    dual-channel recording only when
  each controlled dispatch also carries a valid consent reference; otherwise the call
  is blocked before dialing.
5. Keep `controlled_test_mode` enabled for controlled tests. Before disabling it, add a
   legally reviewed timezone and calling window.
6. Add regression scenarios and run the full suite.

Optional `personalized_preamble` configuration contains exactly three question
templates: `recipient_confirmation` with `{recipient_name}`, `property_timing` with
`{property_reference}`, and `qualification` without placeholders. The first template
must include all required AI/company disclosures. Metadata is opt-in; without it, normal
`opening`/`opening_variants` behavior is unchanged.

Campaign loading rejects undeclared outcomes, missing closings/questions/types,
unsupported field types, invalid retention/attempt settings, and introductions missing
required disclosures. It also rejects prohibited or human-identity claims in every
directly spoken campaign surface: introductions, openings, variants, questions,
closings, transfer/voicemail/error messages, and FAQ answers.

## Where to Change What

Choose the narrowest owning surface. Keep campaign-specific behavior in campaign JSON;
change Python only when the behavior is reusable across campaigns or must be enforced
outside the model.

| Need | Primary files | How to change it | Change it when |
|---|---|---|---|
| Change company identity, goal, wording, fields, FAQs, flow, voice style, or limits | `campaigns/*.json` | Edit configuration, keep a unique `campaign_id`, then run campaign and conversation tests | The behavior belongs to one campaign and fits the existing schema |
| Add or change campaign schema | `src/speaking_agent/campaign.py`, every campaign JSON, `tests/test_campaign.py` | Add strict parsing/defaults and reject malformed or unsafe values | Multiple campaigns need a new declarative capability |
| Change DNC, transfer, outcome evidence, field grounding, or response safety | `src/speaking_agent/policy.py`, `src/speaking_agent/text_safety.py` | Implement deterministic evidence rules and adversarial regressions | The rule affects compliance, persisted state, or terminal actions |
| Change adaptive turn progression or dialogue memory | `src/speaking_agent/conversation.py`, `src/speaking_agent/domain.py` | Preserve policy ownership; update state only from validated evidence | The application must alter what objective/question comes next |
| Change Qwen prompting, context selection, parsing, or budget | `src/speaking_agent/adapters/llm/qwen_mlx.py` | Keep sparse structured output, the 14,500-character cap, newest dialogue, and required policy context | Natural-language planning needs improvement without weakening policy |
| Add another LLM | `src/speaking_agent/model.py`, `src/speaking_agent/adapters/llm/`, composition entry points | Implement `ConversationModel`; translate provider output to `ModelInterpretation` | A platform cannot use MLX or a different quality/latency profile is needed |
| Add or tune ASR/TTS | `src/speaking_agent/speech.py`, `src/speaking_agent/adapters/asr/`, `src/speaking_agent/adapters/tts/`, `src/speaking_agent/delivery.py` | Preserve application PCM/events and cancellation contracts | A new backend, language, voice, or measured quality target requires it |
| Change endpointing, short-speech handling, or local audio | `src/speaking_agent/turn_detection.py`, `src/speaking_agent/local_voice_chat.py`, `src/speaking_agent/adapters/telephony/sounddevice_local.py` | Tune from captured consented scenarios; keep device/sample-rate behavior explicit | Audio evidence shows missed speech, false turns, echo, or latency problems |
| Change call lifecycle, barge-in, transfer, or cleanup | `src/speaking_agent/voice_session.py`, `src/speaking_agent/transport.py` | Keep one `CallSession` owner and bounded cancellation/cleanup | Behavior must be identical across local and LiveKit transports |
| Change LiveKit or controlled outbound SIP | `src/speaking_agent/livekit_worker.py`, `src/speaking_agent/adapters/telephony/livekit_room.py`, `src/speaking_agent/call_cli.py`, `src/speaking_agent/outbound.py` | Keep SDK/SIP types inside adapters and preserve allowlist/suppression gates | Provider behavior or controlled-call requirements change |
| Change market-data integration | `src/speaking_agent/market_data.py`, composition entry points | Preserve the normalized actual-transaction/current-listing distinction and approved HTTPS feeds | A reviewed provider or internal mapping service is available |
| Change lead classification/CRM delivery | `src/speaking_agent/lead_workflow.py`, `src/speaking_agent/livekit_worker.py` | Keep raw numbers transient, use `call_id` idempotency, and preserve P1-P4/task semantics | CRM or notification requirements change |
| Change persistence, privacy, retention, or metrics | `src/speaking_agent/records.py`, `src/speaking_agent/recording.py`, `src/speaking_agent/suppression.py`, `src/speaking_agent/adapters/storage/sqlite.py`, `src/speaking_agent/metrics.py` | Migrate explicitly, keep raw numbers out, gate transcripts by campaign, and preserve atomic attempt checks | The structured record contract or reviewed retention policy changes |
| Change consented audio capture or recording retention | `src/speaking_agent/audio_recording.py`, `src/speaking_agent/voice_session.py`, transports, `src/speaking_agent/retention_worker.py` | Preserve explicit per-call consent, private dual-channel artifacts, transport-confirmed playout, and expiry manifests | A reviewed quality/training workflow changes |
| Change operator workflows | `src/speaking_agent/operator_cli.py`, `src/speaking_agent/retention_worker.py` | Add commands over repository interfaces rather than direct SQL | Operators need a repeatable inspection or maintenance action |

## How to Extend Safely

1. Start with a failing scenario or adapter-contract test in the matching `tests/test_*.py`.
2. Prefer a campaign-only edit when the schema can express the requirement.
3. Put critical decisions in policy/application code; treat model output as untrusted
  suggestions.
4. Add a protocol or abstraction only when a second implementation or real duplication
  requires it.
5. Run focused tests after the first edit, then the complete quality gate above.
6. For audio, model, LiveKit, SIP, or platform work, also run the native integration on
  the target hardware. Wheel installation or mocked tests do not establish support.
7. Update this README for user-visible behavior and update `ROADMAP.md` when an item is
  completed, split, reprioritized, or newly discovered.

Minimum validation by change type:

| Change type | Required evidence |
|---|---|
| Campaign content/schema | Campaign tests plus relevant conversation scenarios |
| Policy/state/lifecycle | Focused regression, full suite, and failure-path coverage |
| LLM/ASR/TTS adapter | Contract tests plus one real model smoke on supported hardware |
| Local audio/turn detection | Unit tests plus microphone/speaker round trip on the target device |
| LiveKit/SIP | Adapter tests plus controlled room/call evidence; never use an unapproved number |
| Storage/privacy | Migration, concurrency, retention, and no-sensitive-data assertions |
| New platform/profile | Clean native install, full gate, hardware round trips, and published measurements |

## Adapter Replacement

- LLM: implement `ConversationModel.prepare`, `interpret`, and `close`; translate output
  to `ModelInterpretation`.
- ASR: implement `SpeechRecognizer`; emit cumulative partial/final `TranscriptEvent`
  objects.
- TTS: implement `SpeechSynthesizer`; emit signed 16-bit little-endian PCM
  `AudioFrame` objects and support cancellation.
- Telephony: implement `CallTransport`, including queue clearing and playout draining.
- Storage: implement `CallRepository`, including suppression, attempt, and retention
  operations.

Mocks cover every boundary. They are the supported path for testing without models,
audio hardware, LiveKit, or telephony.

## Known Limitations

- LiveKit room audio is verified locally in both directions. PSTN integration is
  implemented but not exercised because no project credentials, SIP trunk, or controlled
  number were supplied.
- Qwen3-ASR currently receives a completed detected utterance, then streams decoder
  events. It does not incrementally encode live PCM as it arrives.
- The included energy turn detector is intentionally simple. Real telephone noise should
  be measured before selecting or tuning a stronger VAD/noise-cancellation adapter.
- Answering-machine detection uses explicit machine phrases and otherwise defaults to
  human to avoid discarding long human responses. Production AMD needs measured data.
- MLX cancellation is cooperative. Teardown waits for a bounded grace period and
  quarantines a stalled adapter; a native Python worker thread may finish later.
- Human-identity protection combines campaign-load validation and runtime regex-based
  filtering. Known variants are covered, but new paraphrases should be added as
  adversarial tests; truthful AI/automation disclosure must never be weakened.
- The Qwen prompt is capped at 14,500 characters by dropping oldest dialogue and then
  optional guidance. Required policy/state/current-turn context is never silently
  dropped; an irreducible oversized campaign fails explicitly and should be shortened.
- Campaign wording and question flow are configurable, but deterministic extraction and
  outcome grounding currently include property-domain logic in `policy.py`. Introduce a
  tested domain-policy boundary before deploying a materially different campaign type.
- Greedy Qwen TTS removes sampling variance and repeated-text output is deterministic on
  the verified checkpoint. Perceptual consistency across different sentences still
  requires listening benchmarks; use `--tts-temperature 0` for the most stable profile.
- Consented audio recording stores sensitive raw PCM and a minimal manifest. Application-
  level encryption, de-identification, annotation, approval workflow, and training-data
  export are not implemented; owner-only permissions and retention are not substitutes
  for those production controls.
- SoundDevice reports callback-consumed agent PCM precisely. LiveKit currently reports
  completed playout batches; when a LiveKit turn is interrupted, its agent-channel prefix
  may be omitted because the SDK path does not expose an exact played-sample cursor.
- The example company, wording, retention, and policy values are demonstrations, not
  legal advice. Production use requires authoritative review for identity, consent,
  recording, calling times, retention, transfer, and suppression rules.
- No queue, broker, microservice split, retry farm, or multi-machine deployment is added;
  the current requirement is one process and one controlled call.
- DLD/Property Monitor/listing credentials and exact identifier mappings were not
  supplied, so live market accuracy and external webhook delivery remain unverified.