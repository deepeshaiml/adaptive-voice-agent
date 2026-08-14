# Role

Act as a **senior software architect, real-time systems engineer, AI/LLM engineer, voice AI engineer, and implementation engineer**.

Design and incrementally implement a **real-time AI voice calling platform**.

The system will place telephone calls to property owners and conduct natural conversations to determine whether they are interested in:

* selling their property;
* renting/leasing their property;
* considering either later;
* requesting a callback;
* speaking with a human agent;
* or not being interested.

The AI must understand free-form human responses and **adapt the conversation dynamically rather than following a rigid question-by-question script**.

The system must remain transparent about required company/caller identification and must not falsely claim to be a human.

---

# Development Environment

Initial development environment:

* Apple M5 Pro
* 48 GB unified memory
* macOS
* VS Code
* GitHub Copilot Agent Mode
* Python preferred unless another language has a concrete technical advantage

Initial target is **one local development machine and one concurrent test call**.

Do not design for 1,000 simultaneous calls initially.

The architecture must allow scaling later without redesigning the domain/application core.

---

# Primary Technical Direction

Initially evaluate and implement around:

* LiveKit Agents for real-time voice orchestration
* LiveKit SIP/telephony abstraction
* streaming speech-to-text
* local or replaceable LLM inference
* streaming text-to-speech
* Apple Silicon / MLX where practical
* Qwen3-ASR as an initial ASR candidate
* Qwen-family local LLM as an initial reasoning candidate
* Qwen3-TTS as an initial TTS candidate

These are **initial adapters, not permanent architectural dependencies**.

The core application must not depend directly on LiveKit, Qwen, MLX, Twilio, Telnyx, any database vendor, or any particular AI provider.

All such systems must be behind interfaces/adapters.

Before choosing versions, APIs, or dependencies, verify them against current official documentation when research tools are available.

Never invent a version number or API.

If current information cannot be verified, state that clearly.

---

# Core Engineering Principles

Prioritize, in this order:

1. Correctness
2. Simplicity
3. Low conversational latency
4. Maintainability
5. Modularity
6. Testability
7. Portability
8. Observability
9. Scalability

Build a system that is:

* modular;
* decoupled;
* minimal;
* beginner-readable;
* hardware-independent where practical;
* OS-independent where practical;
* vendor-replaceable;
* testable without telephony;
* testable without AI models;
* runnable locally;
* scalable later.

> Start simple. Keep boundaries clean. Add complexity only when an actual requirement demands it.

---

# Architecture Principle

Prefer a modular monolith initially.

Conceptually:

```text
Telephony / Audio
        ↓
Voice Session
        ↓
Conversation Application
        ↓
Conversation Policy
        ↓
Domain
        ↓
Lead Qualification
        ↓
Result / CRM

External systems connect through interfaces/adapters.
```

Use dependency inversion:

```text
Domain / Application
        ↓
Interfaces
        ↑
Adapters
```

The domain and conversation policy must not import:

* LiveKit
* SIP libraries
* MLX
* Qwen SDKs
* database drivers
* HTTP frameworks
* cloud SDKs
* operating-system-specific APIs

---

# Critical Requirement: Script-Driven Adaptive Conversation

The system must accept a **call campaign / speaking script as runtime configuration**.

DO NOT hardcode a particular real-estate script into application logic.

A campaign script represents:

* objective;
* identity/company introduction;
* opening;
* required disclosures;
* desired outcomes;
* qualifying questions;
* optional questions;
* objection guidance;
* answers to common questions;
* prohibited statements;
* required information;
* escalation conditions;
* human-transfer conditions;
* callback rules;
* conversation-ending rules;
* output fields.

The runtime AI must treat this as a **conversation policy**, not a literal screenplay.

For example, configuration may define:

```yaml
campaign:
  name: property-owner-qualification

  objective:
    Determine whether the property owner is interested in:
      - selling
      - renting
      - future consideration
      - callback
      - neither

  opening:
    identify_company: true
    purpose: property enquiry

  desired_outcomes:
    - SELL
    - RENT
    - FUTURE
    - CALLBACK
    - HUMAN_TRANSFER
    - NOT_INTERESTED
    - WRONG_NUMBER
    - DO_NOT_CONTACT
    - UNKNOWN

  required_fields:
    - intent

  seller_fields:
    - property_location
    - property_type
    - expected_price
    - selling_timeline
    - currently_listed

  rental_fields:
    - property_location
    - property_type
    - availability_date
    - expected_rent

  behavior:
    ask_one_question_at_a_time: true
    avoid_repeating_known_information: true
    allow_interruptions: true
    concise_responses: true
```

The exact schema may evolve.

Keep it simple initially.

---

# Adaptive Conversation Requirements

The AI must **not mechanically follow the script from top to bottom**.

It must:

* understand free-form responses;
* recognize information volunteered early;
* skip questions whose answers are already known;
* change branches according to user intent;
* answer relevant questions;
* return naturally to the objective;
* handle interruptions;
* handle corrections;
* handle uncertainty;
* handle objections;
* allow the caller to change their mind;
* avoid repeatedly asking the same question;
* ask one primary question at a time;
* keep telephone responses relatively short;
* gracefully terminate when the person is not interested.

Example:

```text
Agent:
Are you considering selling or renting your property?

Owner:
I actually already listed my apartment for sale last week.

System should infer:

intent = SELL
currently_listed = true

Do NOT ask:
"Are you considering selling?"

Continue with the next useful qualification question.
```

Another example:

```text
Agent:
Would you consider selling?

Owner:
How did you get my number?

Agent:
Answer the question according to configured company policy.

Then, if appropriate, return naturally to the conversation objective.

Do not ignore the owner's question.
```

---

# Hybrid Conversation Architecture

Use a hybrid design.

The **LLM controls natural language understanding and generation**.

Deterministic application logic controls:

* call lifecycle;
* compliance rules;
* hard stop conditions;
* required disclosures;
* do-not-contact handling;
* maximum retries;
* human transfer;
* tool permissions;
* persistence;
* final outcome;
* validation.

Do NOT make the LLM solely responsible for critical application state.

Suggested conceptual states:

```text
CREATED
   ↓
DIALING
   ↓
CONNECTED
   ↓
OPENING
   ↓
DISCOVERY
   ↓
QUALIFICATION
   ↓
optional:
OBJECTION / CALLBACK / TRANSFER
   ↓
CLOSING
   ↓
COMPLETED
```

Do not create an excessively rigid state machine.

Conversation state and business state are different concerns.

---

# Structured Conversation State

Maintain structured state independent of chat history.

Example:

```json
{
  "call_id": "...",
  "campaign_id": "...",
  "phone_number": "...",
  "intent": "SELL",
  "property_location": "Dubai Marina",
  "property_type": "2 bedroom apartment",
  "selling_timeline": "1-3 months",
  "currently_listed": false,
  "callback_requested": true,
  "human_transfer_requested": false,
  "do_not_contact": false,
  "confidence": {},
  "conversation_stage": "QUALIFICATION"
}
```

Do not rely on the entire transcript as the database.

Keep:

```text
Transcript ≠ State
```

The transcript is evidence/history.

Structured state represents what the application currently knows.

---

# LLM Responsibilities

Create an interface such as:

```text
ConversationModel
```

Its responsibility may include:

* interpret latest utterance;
* extract structured information;
* identify intent;
* choose conversational action;
* generate a short response;
* request a tool/action when needed.

Do not let provider-specific tool-call formats leak into the application/domain layer.

Use application-owned types.

---

# Speech-to-Text Interface

Create a replaceable abstraction such as:

```text
SpeechRecognizer
```

It should support, where available:

* streaming audio input;
* partial transcripts;
* final transcripts;
* cancellation;
* language metadata;
* timing metadata;
* errors.

Initial candidate:

```text
Qwen3-ASR adapter
```

Also create:

```text
MockSpeechRecognizer
```

so the entire conversation system can run without an ASR model.

---

# Text-to-Speech Interface

Create an abstraction such as:

```text
SpeechSynthesizer
```

Support:

* streaming output;
* cancellation;
* interruption;
* voice selection;
* speaking style where supported.

Initial candidate:

```text
Qwen3-TTS adapter
```

Also implement:

```text
MockSpeechSynthesizer
```

and preferably a simple system TTS adapter for debugging.

---

# LLM Interface

Create an abstraction such as:

```text
ConversationModel
```

Initial candidate:

```text
local Qwen model through MLX
```

But keep it replaceable.

Implement:

```text
MockConversationModel
```

for deterministic testing.

---

# Telephony Interface

Create an abstraction such as:

```text
CallTransport
```

Responsibilities:

* dial;
* answer/connected notification;
* incoming audio stream;
* outgoing audio stream;
* hang up;
* call status events;
* transfer where supported.

Initial adapter:

```text
LiveKit
```

SIP/provider-specific concerns must remain outside the core.

---

# Real-Time Requirements

This is a real-time interactive system.

Measure latency instead of assuming performance.

Record timing for at least:

```text
speech_end
ASR_partial
ASR_final
LLM_start
LLM_first_token
TTS_start
TTS_first_audio
playback_start
```

Derive:

```text
user speech end → first agent audio
```

as a critical metric.

Do not optimize prematurely, but design streaming boundaries correctly from the beginning.

---

# Interruptions / Barge-In

The person must be able to interrupt the AI.

On a confirmed user interruption:

1. stop/cancel current TTS;
2. stop outgoing buffered speech where possible;
3. continue listening;
4. understand the interruption;
5. respond to what was actually said.

Do not treat every tiny sound as an interruption.

Allow the voice/transport layer to distinguish genuine interruptions from:

* background noise;
* short acknowledgements;
* accidental sounds.

Keep turn-detection policy configurable.

---

# Concurrency and Cancellation

Use structured concurrency.

Every call should have an isolated:

```text
CallSession
```

A session owns resources related to that call.

When a call ends:

* cancel ASR;
* cancel LLM generation;
* cancel TTS;
* stop audio;
* flush required persistence;
* release resources.

Avoid orphaned background tasks.

---

# Outbound Call Lifecycle

Support results such as:

```text
ANSWERED_HUMAN
VOICEMAIL
NO_ANSWER
BUSY
FAILED
REJECTED
INVALID_NUMBER
```

Voicemail handling should be separate from human conversation behavior.

Do not assume every connected call is a human.

---

# Lead Outcome

At the end of every completed conversation produce a structured result.

Example:

```json
{
  "outcome": "SELL",
  "qualified": true,
  "summary": "Owner is considering selling within approximately two months.",
  "fields": {
    "property_location": "Dubai Marina",
    "property_type": "2BR apartment",
    "timeline": "1-3 months"
  },
  "callback_requested": true,
  "human_followup_required": true
}
```

The application should be able to persist this regardless of which AI/telephony provider is being used.

---

# Safety and Business Rules

Business-critical rules must be enforceable outside the LLM.

Examples include:

* do-not-contact requests;
* explicit refusal;
* maximum call attempts;
* permitted call times;
* required caller/company identification;
* transfer authorization;
* data retention policy;
* recording configuration;
* campaign enable/disable.

Implement these as policies/configuration, not prompt text alone.

Do not invent jurisdiction-specific legal requirements.

Where legal/compliance requirements are needed, mark them as configuration and require verification from authoritative sources before production use.

---

# Natural Conversation

The desired experience is:

* natural;
* concise;
* responsive;
* warm but professional;
* interruption-friendly;
* context-aware.

Do not intentionally add random filler words or artificial mistakes solely to deceive the recipient into believing the AI is human.

Do not make every response a full paragraph.

Telephone responses should generally be short unless the caller asks for an explanation.

Support natural acknowledgement where appropriate without overusing:

```text
okay
got it
sure
understood
```

Conversation quality should come from:

* good latency;
* appropriate phrasing;
* remembering context;
* interruption handling;
* good speech synthesis;
* correct turn-taking;

not from fake human errors.

---

# Script Test Harness

Build a hardware-free conversation simulator early.

It should allow tests like:

```text
Owner: I'm not interested.
Expected:
  outcome = NOT_INTERESTED
  conversation ends
```

```text
Owner: I might sell in two months.
Expected:
  intent = SELL
  selling_timeline populated
  ask next missing relevant question
```

```text
Owner: I'm renting it already.
Expected:
  system understands context
  does not blindly ask the original question again
```

Allow complete simulated conversations using text only.

This simulator must use the same conversation/application logic as the real telephone system.

---

# Testing Strategy

Prioritize:

```text
Unit
  ↓
Conversation scenario tests
  ↓
Adapter integration tests
  ↓
Local audio tests
  ↓
LiveKit tests
  ↓
Single real telephone call
  ↓
Limited controlled pilot
```

Create regression conversation scenarios.

Test:

* happy path;
* sell;
* rent;
* future;
* not interested;
* wrong number;
* callback;
* interruption;
* objection;
* user question;
* repeated information;
* malformed ASR;
* LLM timeout;
* TTS failure;
* dropped call.

---

# Observability

Use structured logs.

Every call should have:

```text
call_id
campaign_id
session_id
```

Track metrics such as:

* calls attempted;
* answered;
* human answered;
* voicemail;
* qualified;
* sell leads;
* rent leads;
* callbacks;
* not interested;
* failures;
* average conversation duration;
* ASR latency;
* LLM latency;
* TTS latency;
* end-to-first-audio latency.

Do not log unnecessary personal or sensitive information.

---

# Persistence

Start with the simplest appropriate storage.

For development, SQLite is acceptable if it satisfies requirements.

Access persistence through interfaces.

The application must be replaceable later with PostgreSQL or another appropriate database without rewriting conversation/domain logic.

Do not introduce Redis, Kafka, or a message broker unless a measured requirement demands one.

---

# Scalability

Start with:

```text
One machine
   ↓
One application process
   ↓
One call
```

Then:

```text
Multiple concurrent CallSession instances
```

Only later, when required:

```text
Multiple workers
      ↓
Job coordination
      ↓
Multiple machines
```

Do not implement distributed architecture for the initial prototype.

Design clean boundaries that make it possible later.

---

# Recommended Initial Project Structure

Keep it small.

For example:

```text
src/
  domain/
  application/
  conversation/
  interfaces/
  adapters/
    asr/
    tts/
    llm/
    telephony/
    storage/
  config/
  observability/

tests/
  unit/
  conversations/
  integration/

campaigns/
  property_owner.yaml
```

Adjust this structure if a simpler arrangement is justified.

Do not create empty folders merely to match the diagram.

---

# Implementation Strategy

DO NOT attempt the entire production system in one uncontrolled implementation pass.

Work incrementally.

## Phase 1 — Conversation Core

Build:

* campaign configuration;
* conversation state;
* conversation policy;
* structured outcomes;
* mock LLM;
* text-only simulator;
* unit/scenario tests.

Must run without:

* LiveKit;
* ASR;
* TTS;
* Internet;
* database.

STOP and verify tests.

## Phase 2 — Local LLM

Add the replaceable local LLM adapter.

Verify adaptive conversation using text.

STOP and test.

## Phase 3 — Local Speech

Add ASR and TTS adapters.

Test locally using microphone/speaker or audio files.

Measure latency.

STOP and test.

## Phase 4 — Real-Time Voice Session

Integrate LiveKit.

Implement:

* streaming;
* turn detection;
* interruptions;
* cancellation;
* call session lifecycle.

STOP and test locally.

## Phase 5 — Telephony

Add SIP/outbound calling.

First call only an explicitly controlled test number.

Verify:

* connect;
* human answer;
* conversation;
* barge-in;
* hangup;
* outcome persistence.

STOP and test.

## Phase 6 — Persistence and Operator View

Add the minimum required persistence and lead inspection functionality.

## Phase 7 — Production Hardening

Only after measured need, add:

* retries;
* concurrency control;
* rate limiting;
* job queue;
* deployment infrastructure;
* expanded observability;
* recovery.

---

# Working Method in Agent Mode

Before modifying files:

1. inspect the repository;
2. identify what already exists;
3. state assumptions;
4. create a concise implementation plan;
5. select the smallest useful vertical slice.

During implementation:

* make small coherent changes;
* run tests after meaningful changes;
* inspect failures;
* fix root causes;
* do not hide errors;
* do not disable tests to make them pass;
* do not leave placeholder implementations unless explicitly identified.

Do not rewrite working code unnecessarily.

If requirements are uncertain, prefer the simplest reversible design.

---

# Technology Selection

Prefer:

1. stable;
2. actively maintained;
3. open source;
4. Apple-Silicon compatible where practical;
5. portable;
6. simple;
7. well documented;
8. testable;
9. performant.

Do not automatically choose the newest technology.

Verify compatibility before implementing it.

Experimental/community adapters are allowed for a prototype when clearly isolated behind an interface.

---

# Explain Important Decisions

For important architectural choices briefly explain:

* what was selected;
* why;
* simplest alternative;
* trade-off;
* how to replace it;
* how it scales later.

Avoid lengthy theoretical explanations when code or a test provides a clearer answer.

---

# Required Documentation

Maintain concise documentation containing:

* architecture;
* how to run;
* how to test;
* how campaigns/scripts work;
* how to create a new campaign;
* how to swap ASR;
* how to swap LLM;
* how to swap TTS;
* how to test without telephony;
* how to make a controlled test call;
* known limitations.

---

# Final Quality Gate

Before considering any phase complete verify:

* [ ] Simplest practical implementation
* [ ] Beginner-readable
* [ ] No unnecessary abstractions
* [ ] No unnecessary dependencies
* [ ] Conversation policy separated from infrastructure
* [ ] Speaking script stored as runtime configuration
* [ ] Script can be replaced without code changes
* [ ] Conversation adapts instead of reading script literally
* [ ] Structured conversation state exists
* [ ] Transcript is not treated as application state
* [ ] Critical business rules do not depend solely on the LLM
* [ ] ASR replaceable
* [ ] LLM replaceable
* [ ] TTS replaceable
* [ ] Telephony replaceable
* [ ] Storage replaceable
* [ ] Text-only simulator works
* [ ] Unit tests work without external services
* [ ] Conversation scenario tests exist
* [ ] Interruptions/cancellation handled
* [ ] Latency measured
* [ ] Failure paths tested
* [ ] External resources released after a call
* [ ] No premature microservices
* [ ] No premature message broker
* [ ] No unnecessary vendor lock-in
* [ ] Security considered
* [ ] Compliance rules configurable
* [ ] Current dependencies verified rather than guessed

> **Build the smallest working real-time conversational calling system first. Make the conversation engine independent of the speech models and telephony provider. Treat the speaking script as configurable policy, not hardcoded dialogue.**
