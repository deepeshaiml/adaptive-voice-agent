# Optimization and Portability Roadmap

Last updated: **2026-08-16**

Checklist meaning:

- `[x]` is implemented and verified in the current supported scope.
- `[ ]` is incomplete; partial foundations are described explicitly and must not be
      interpreted as platform support.

Current validation baseline: **161 unit/scenario/adapter tests**, clean `pip check`,
clean bytecode compilation, clean `git diff --check`, resolved local Markdown links,
and no editor diagnostics. A single warm local Qwen planning smoke completed in 1.214
seconds. That number is useful as a regression signal only; repeatable P50/P95
benchmarks remain required.

## Current Baseline

The complete local inference path is currently verified on Apple-silicon macOS using
MLX/Metal. The conversation core, storage, and provider protocols are portable Python,
but the shipped LLM, ASR, and TTS implementations are MLX-specific. Installability on a
platform does not yet imply supported or optimized inference.

| Target | Current status |
|---|---|
| macOS ARM64 | Verified local LLM, ASR, TTS, audio, and LiveKit transport |
| macOS x86_64 | Core only; no supported local MLX inference |
| Linux x86_64 CPU | Core and LiveKit are candidates; local inference not integrated |
| Linux x86_64 NVIDIA CUDA | Next accelerated target; backend selection and native evidence remain open |
| Linux ARM64 | Core and LiveKit are candidates; local inference not integrated |
| Windows x86_64 | Core, audio, and LiveKit are candidates; local inference not integrated |
| Windows ARM64 | Core candidate; LiveKit and local inference gaps remain |
| AMD or Intel GPU | No accelerated inference backend yet |

## Progress Snapshot

| Capability | Current progress | Primary ownership | Remaining work |
|---|---|---|---|
| Conversation core | Adaptive campaign flow, structured state, grounded fields, corrections, callbacks, transfers, and terminal outcomes are implemented | `conversation.py`, `policy.py`, `domain.py` | Recorded scenario benchmark and broader language/accent evidence |
| Conversation memory | Bounded two-sided in-memory history tracks pending, interrupted, and delivered agent speech without persistence | `conversation.py`, `voice_session.py`, `domain.py` | Long-call quality benchmark and summarization only if measured need appears |
| Model planning | Sparse structured Qwen planning, relevant guidance selection, exact FAQ fast path, and 14,500-character prompt cap are implemented | `adapters/llm/qwen_mlx.py` | Stable-prefix caching, benchmark harness, larger/portable model comparison |
| Safety/compliance | Deterministic DNC/callback/transfer precedence, field grounding, disclosure enforcement, and identity-claim filtering are implemented | `campaign.py`, `policy.py`, `text_safety.py` | Authoritative jurisdiction review and continuing adversarial paraphrase tests |
| Speech | Local Qwen ASR/TTS, campaign vocabulary, campaign voice style, short utterances, cancellation, and WAV diagnostics are implemented | `speech.py`, `adapters/asr/`, `adapters/tts/`, `delivery.py` | Incremental ASR input, semantic endpointing, broader language/voice evaluation |
| Local audio | Half duplex, speaker mode, and experimental full duplex with barge-in/output-reference echo suppression are implemented | `local_voice_chat.py`, `turn_detection.py`, `adapters/telephony/sounddevice_local.py` | Production AEC/noise suppression/VAD and measured room/device matrix |
| LiveKit/SIP | Bidirectional room audio and controlled outbound dispatch are implemented | `livekit_worker.py`, `adapters/telephony/livekit_room.py`, `call_cli.py` | Real controlled PSTN call, trunk-specific transfer, voicemail, and failure evidence |
| Persistence/privacy | SQLite WAL, atomic attempt reservation, keyed suppression, retention, structured records, and metrics are implemented | `adapters/storage/sqlite.py`, `suppression.py`, `recording.py`, `metrics.py` | Production database adapter only when concurrency/deployment measurements require it |
| Portability | Application protocols and core are portable | `model.py`, `speech.py`, `transport.py`, repository interfaces | Runtime profile selection and native non-Apple inference/audio validation |

## Definition of Supported

A target is supported only when all applicable checks pass on native hardware:

- clean installation from documented dependency groups;
- automated unit and integration tests;
- microphone-to-ASR and TTS-to-speaker round trips;
- deterministic campaign, hard-stop, and persistence behavior;
- real LiveKit bidirectional audio where the SDK supports the target;
- measured cold start, memory use, real-time factor, and speech-end-to-first-audio latency;
- explicit accelerated backend reporting, with no silent CPU or mock fallback;
- documented hardware, driver, OS, and model-memory requirements.

## P0: Platform Detection and Packaging

Current foundation: package extras already separate `mlx`, `speech`, `local`,
`livekit`, and `realtime`, but they are feature groups rather than tested platform
profiles.

- [ ] Add an `adaptive-voice-doctor` command that reports OS, architecture, CPU, GPU,
      available acceleration, audio devices, model backend, and missing requirements.
- [ ] Fail fast with actionable errors for unsupported OS/backend combinations.
- [ ] Split dependency groups by runtime profile instead of installing MLX everywhere:
      Apple MLX, Linux CPU, Linux CUDA 12/13, portable CPU, LiveKit, and development.
- [ ] Add reproducible lock files or constraints for each supported profile.
- [ ] Add CI for Linux x86_64, Windows x86_64, and macOS ARM64; add native ARM runners
      where hosted runners are unavailable.
- [ ] Publish the tested compatibility matrix in the README for every release.

## P1: Portable Inference Backends

Keep application protocols unchanged and select adapters through one composition layer.

- [ ] **Apple profile:** retain MLX/Metal and benchmark model/quantization choices.
- [ ] **Linux NVIDIA profile:** verify current upstream MLX platform support rather than
      assuming CUDA compatibility; if unsupported, evaluate native CUDA and portable
      alternatives per workload before choosing production backends.
- [ ] **Portable LLM profile:** add a llama.cpp/GGUF or equivalent backend supporting
      CPU, Metal, CUDA, and other available accelerators.
- [ ] **Portable ASR profile:** evaluate whisper.cpp, faster-whisper, and ONNX options;
      choose per target based on latency and accuracy, not API similarity.
- [ ] **Portable TTS profile:** add a CPU-capable Piper/Kokoro/ONNX adapter; optionally
      use an upstream CUDA Qwen TTS adapter where quality justifies the hardware cost.
- [ ] **Remote profile:** add provider-neutral HTTP adapters for deployments where local
      inference is unavailable or undesirable.
- [ ] Add explicit `--profile`/configuration selection and log the selected backend,
      model, precision, device, and fallback policy at startup.

## P2: Cross-Platform Audio and Real-Time Behavior

- [x] Add experimental local full-duplex streaming with output-reference echo
      suppression and `CallSession` barge-in.
- [ ] Open devices at supported native sample rates and resample at adapter boundaries;
      do not assume every device accepts 16 kHz input and 24 kHz output.
- [ ] Document and test CoreAudio, WASAPI, and ALSA/PulseAudio/PipeWire device behavior.
- [ ] Add acoustic echo cancellation, noise suppression, and a production VAD.
- [ ] Validate full-duplex local conversation and publish measured barge-in performance
      without headphones across representative rooms and devices.
- [ ] Replace completed-utterance ASR with incremental audio encoding where supported.
- [x] Preserve bounded cancellation and cleanup across the current MLX, SoundDevice,
      and LiveKit implementations.
- [ ] Re-run cancellation, cleanup, and stalled-native-worker tests for every new native
      backend before declaring that backend supported.

## P3: Performance Engineering

- [ ] Create a repeatable benchmark command and machine-readable result format.
- [ ] Measure model download/load time, warm-up, peak RAM/VRAM, ASR real-time factor,
      token latency, TTS real-time factor, and end-to-end response latency.
- [ ] Establish per-profile P50/P95 latency and memory budgets from measured hardware.
- [ ] Preload models and warm kernels before accepting calls.
- [ ] Evaluate prompt/KV caching for the stable campaign prefix.
- [ ] Tune quantization per target instead of assuming one precision is optimal.
- [x] Bound prompt context and generation lengths with deterministic oldest-first
      dialogue trimming and explicit irreducible-overflow failure.
- [ ] Avoid rebuilding invariant campaign and policy prompt data on every turn.
- [ ] Tune CPU thread counts, affinity, CUDA streams, and memory pools where supported.
- [ ] Select model sizes automatically from available RAM/VRAM only with an explicit,
      observable policy and operator override.

## P4: Conversational Intelligence and Naturalness

- [x] Provide bounded, in-memory two-sided dialogue history and structured call state to
      the conversation model without persisting transcripts.
- [x] Add a structured response planner/realizer contract that can vary transitions and phrasing
      while application policy still owns hard stops, fields, and the next objective.
- [x] Track whether planned agent speech is pending, interrupted, or delivered so later
      turns never assume unheard wording was completed.
- [x] Ground model-proposed outcomes and field updates in deterministic utterance
      evidence, including cue-local corrections and punctuation-free sell/rent pivots.
- [x] Select relevant campaign guidance, scenarios, FAQs, voice style, opening variants,
      and ASR vocabulary without passing the complete campaign on every turn.
- [x] Enforce permanent DNC, temporary callback, transfer, disclosure, and prohibited
      identity rules outside the model, with campaign-load and runtime defenses.
- [ ] Move property-specific deterministic extraction/grounding behind a declarative or
      pluggable domain-policy boundary before adding non-property campaigns; never
      replace deterministic validation with model trust alone.
- [ ] Evaluate larger instruction models against the 4B baseline for reference
      resolution, correction handling, question answering, and hallucination rate.
- [ ] Stream safe response text into TTS by sentence instead of waiting for complete JSON
      generation, with a measured first-audio target.
- [ ] Add semantic endpointing so short pauses do not end a thought and long silence does
      not add unnecessary delay.
- [ ] Add natural backchannel timing and prosody controls without fake hesitations,
      identity deception, or repetitive acknowledgements.
- [ ] Build recorded, consented scenario benchmarks for interruptions, corrections,
      topic changes, ambiguity, accents, noise, and multi-turn recall.

## Next Execution Plan

Work in this order unless measured evidence changes the priority.

| Priority | Work | Where to change | How to implement | Start when | Done when |
|---|---|---|---|---|---|
| 1 | Repeatable benchmark command | Add a benchmark CLI under `src/speaking_agent/`; reuse `observability.py`, `recording.py`, and adapter protocols | Emit JSON with hardware/profile metadata, cold/warm load, RAM, real-time factors, and per-stage latency; keep raw audio/transcripts opt-in and consented | Before further model/audio optimization | Repeated runs produce comparable P50/P95 results and documented hardware metadata |
| 2 | Production audio front end | `turn_detection.py`, `adapters/telephony/sounddevice_local.py`, `local_voice_chat.py`, then `voice_session.py` only for transport-neutral behavior | Evaluate maintained AEC/noise-suppression/VAD libraries against captured consented room scenarios; open native device rates and resample at adapter boundaries | After the benchmark harness can measure barge-in and speech-end latency | No self-interruption or missed near-end speech across the agreed device/room matrix and cleanup tests remain bounded |
| 3 | Controlled PSTN verification | `call_cli.py`, `livekit_worker.py`, `adapters/telephony/livekit_room.py`, campaign calling policy | Use only an allowlisted controlled number and private credentials; verify human, voicemail, no-answer, transfer-unavailable, DNC persistence, and teardown paths | When a reviewed trunk, credentials, and controlled number are available | One documented controlled call matrix passes without exposing credentials or raw numbers |
| 4 | Runtime doctor and profiles | Add doctor/profile composition modules; update `pyproject.toml`, `local_voice_chat.py`, `livekit_worker.py`, and README | Detect OS/architecture/devices/accelerators, select an explicit profile, and fail with actionable errors rather than silently falling back | Before adding the first non-Apple inference backend | Doctor output is deterministic and each startup reports backend, model, precision, device, and fallback policy |
| 5 | Linux NVIDIA backend evaluation | New adapters under `adapters/llm/`, `adapters/asr/`, and `adapters/tts/`; keep `model.py`, `speech.py`, and `transport.py` unchanged | Verify current upstream MLX support first; benchmark it only if supported, then compare native CUDA/portable alternatives independently for LLM, ASR, and TTS | After profiles and benchmark tooling exist on native Linux NVIDIA hardware | Clean install, full tests, real speech round trips, LiveKit audio, explicit GPU use, and published latency/memory results pass |
| 6 | Portable CPU/Linux/Windows profiles | Same adapter boundaries plus CI and constraints/lock files | Add portable implementations one workload at a time; never report support from wheel availability alone | After the Linux NVIDIA composition path proves replaceability | Native matrix in validation order passes all applicable support criteria |
| 7 | Domain-policy boundary | `campaign.py`, `policy.py`, `conversation.py`, and domain-specific tests | Define application-owned deterministic extractors/validators selected explicitly by campaign type; keep shared DNC/disclosure/transfer policy central | Before the first materially non-property campaign | Existing property scenarios remain green and a second domain works without property branches in shared policy |
| 8 | Conversation quality benchmark | `tests/test_conversation.py`, a new consented scenario corpus outside persisted call records, and benchmark reporting | Score task completion, correction handling, grounded extraction, repetition, hallucination, latency, and interruption recovery | Before changing model size, prompt strategy, or adding backchannels | Baseline and candidate runs are reproducible and regressions have explicit thresholds |
| 9 | Natural delivery improvements | `delivery.py`, TTS adapters, `conversation.py`, and `voice_session.py` | Add measured prosody/backchannel timing and safe sentence-level streaming without allowing unvalidated text to reach TTS | After quality and latency benchmarks can detect regressions | First-audio latency improves while safety, interruption, and repetition suites remain green |

## Change and Update Rules

1. Put campaign-specific goals, wording, fields, FAQs, style, and scenario guidance in
      `campaigns/*.json`; change Python only for reusable capability or deterministic
      enforcement.
2. Add schema in `campaign.py`, application state in `domain.py`, deterministic rules in
      `policy.py`, turn progression in `conversation.py`, and provider behavior only under
      `adapters/` or composition entry points.
3. Add a failing regression before changing safety, state, lifecycle, persistence, or
      cancellation behavior. Model output remains untrusted input.
4. Update `README.md` in the same change when commands, defaults, configuration, user
      behavior, supported platforms, or limitations change.
5. Check an item only after its tests and applicable native evidence pass. Record the
      date, hardware/profile, versions, and benchmark artifact in the pull request or
      release notes.
6. Reprioritize only from measured latency/quality/reliability evidence, a concrete
      deployment requirement, or an authoritative compliance requirement.

## P5: Deployment Profiles

- [ ] Provide Linux CPU and NVIDIA CUDA container images with health checks.
- [ ] Keep local desktop audio outside containers unless device forwarding is tested.
- [ ] Add model-cache configuration, offline startup, checksum verification, and disk
      capacity checks.
- [ ] Add graceful readiness: process health is not ready until required models and
      audio/transport dependencies are prepared.
- [ ] Document one-machine desktop, Linux GPU server, and remote-inference deployments.

## P6: Validation Order

1. Linux x86_64 with NVIDIA CUDA.
2. Linux x86_64 CPU fallback.
3. Windows x86_64 with CPU and available GPU acceleration.
4. Linux ARM64 CPU.
5. Windows ARM64 and additional GPU vendors after dependency gaps close.
6. macOS x86_64 as remote/portable CPU only unless a maintained accelerated backend is
   available.

No target should be marked complete from wheel availability alone. Each checkbox requires
native execution evidence and published benchmark results.
