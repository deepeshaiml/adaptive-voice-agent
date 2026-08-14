# Adaptive Voice Agent

A local-first, provider-neutral framework for adaptive voice conversations. It includes
a campaign-driven conversation core, local Qwen LLM/ASR/TTS adapters, microphone and
speaker interaction, a real-time call lifecycle, LiveKit room transport, outbound SIP,
structured SQLite outcomes, operator tooling, and compliance-focused safeguards.

The distribution and repository use the general name `adaptive-voice-agent`. The
internal Python namespace remains `speaking_agent` for compatibility.

A real PSTN call has not been placed from this repository because that requires private
LiveKit credentials, an outbound trunk, and a number explicitly controlled by the
operator. The command is implemented but deliberately blocked without all three.

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
                         v
                  CallRepository ---> SQLite
```

The domain, policy, conversation, and call lifecycle own application types. They do not
import LiveKit, SIP, MLX, Qwen, SQLite, cloud SDKs, or OS-specific APIs. Provider types
are translated only inside adapters.

## Setup

Python 3.11+ is supported. Install the full verified local stack in the configured
virtual environment:

```sh
.venv/bin/python -m pip install -e '.[realtime]'
```

Pinned integration versions are `mlx-lm==0.31.3`, `mlx-audio==0.4.8`,
`livekit-agents==1.6.10`, and its required `livekit==1.1.14`.

Run the complete dependency-free test suite:

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

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

The agent speaks the campaign opening, listens until 700 ms of trailing silence,
prints what Qwen3-ASR heard, runs the real conversation model, and speaks the Qwen3-TTS
response. Each turn reports ASR, LLM, TTS-first-audio, and speech-end-to-response
latency. The local helper applies a warm conversational speaking style by default; use
`--style` to replace it. Audio and transcripts are not written to disk. Press `Ctrl-C`
to stop.

On first use, allow microphone access for VS Code or the terminal in macOS **System
Settings > Privacy & Security > Microphone**. Headphones prevent the microphone from
hearing the agent's speaker output. List and select CoreAudio devices with:

```sh
PYTHONPATH=src .venv/bin/python -m speaking_agent.local_voice_chat --list-devices
PYTHONPATH=src .venv/bin/python -m speaking_agent.local_voice_chat \
  --input-device 2 --output-device 3
```

To test through speakers without headphones, use speaker mode:

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

The helper is intentionally turn-by-turn to make voice, transcription, understanding,
and response quality easy to judge. Barge-in behavior remains covered by `CallSession`
and the LiveKit path rather than this local microphone helper.

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

SQLite defaults to `data/speaking_agent.db`. Records include connection result, answer
kind, lead outcome, validated fields, interruption count, duration, and latency metrics.
They do not contain transcripts or raw telephone numbers. Do-not-contact and attempt
tracking use a keyed HMAC fingerprint. Structured call records and attempt history are
assigned per-row expirations using their campaign's `data_retention_days`; suppression
entries are retained. Run the retention service beside the call worker so expired data is
removed even while no calls are arriving:

```sh
PYTHONPATH=src .venv/bin/python -m speaking_agent.retention_worker \
  --database data/speaking_agent.db --interval-seconds 3600
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

Campaigns are runtime JSON files. The engine contains no real-estate question sequence.
The active local voice campaign is `campaigns/property_owner.json`. That is where you
give the agent its overall goal and flexible scenario guidance:

The repository includes two examples:

- `campaigns/property_owner.json`: detailed adaptive property campaign.
- `campaigns/neoai_property_owner.json`: compact NeoAI-branded campaign.

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
approved facts belong in `faq_answers`; alternate follow-up wording belongs in
`question_variants`.

To create another campaign:

1. Copy `campaigns/property_owner.json` and assign a unique `campaign_id`.
2. Configure the objective, introduction, full opening, and exact disclosure fragments.
3. Define outcomes, outcome guidance, branch fields, field types, and one question per
   field.
4. Configure hard stops, FAQs, prohibited statements, terminal/closing behavior,
   voicemail text, attempt limits, retention, and model failure bounds.
  Set `recording_enabled` explicitly. This prototype supports `false`; `true` blocks the
  call until an audited recording/consent adapter is implemented.
5. Keep `controlled_test_mode` enabled for controlled tests. Before disabling it, add a
   legally reviewed timezone and calling window.
6. Add regression scenarios and run the full suite.

Campaign loading rejects undeclared outcomes, missing closings/questions/types,
unsupported field types, invalid retention/attempt settings, and introductions missing
required disclosures.

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
- The example company, wording, retention, and policy values are demonstrations, not
  legal advice. Production use requires authoritative review for identity, consent,
  recording, calling times, retention, transfer, and suppression rules.
- No queue, broker, microservice split, retry farm, or multi-machine deployment is added;
  the current requirement is one process and one controlled call.