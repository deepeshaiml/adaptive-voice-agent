from __future__ import annotations

from speaking_agent.campaign import Campaign


DEFAULT_VOICE_STYLE = (
    "Speak warmly and naturally with conversational pacing, subtle pauses, and no "
    "announcer tone. Keep the delivery calm and concise."
)


def campaign_voice_style(campaign: Campaign) -> str:
    style = campaign.voice_style
    if not style:
        return DEFAULT_VOICE_STYLE
    ordered_keys = (
        "personality",
        "tone",
        "pacing",
        "verbosity",
        "energy",
        "formality",
        "language",
        "acknowledgement_rule",
    )
    parts = [str(style[key]).strip() for key in ordered_keys if style.get(key)]
    avoid = style.get("avoid")
    if isinstance(avoid, (list, tuple)) and avoid:
        parts.append("Avoid " + ", ".join(str(value) for value in avoid[:6]) + ".")
    return " ".join(parts) or DEFAULT_VOICE_STYLE
