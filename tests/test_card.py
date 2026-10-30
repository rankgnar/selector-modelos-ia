"""Quotes from uncensored model cards (lines copied from real READMEs, 2026-10-30)."""

from selector_ia.finder.card import read_card

HERETIC = """# Qwen3.5-9B Heretic
| **KL divergence** | 0.0793 | 0 *(by definition)* |
| **Refusals** | 6/100 | 100/100 |
Some unrelated line about prompts.
<td>Tool Calling</td>
"""
HAUHAU = """Qwen3.8-27B uncensored by HauhauCS **0/465 Refusals*** .
This is the **Aggressive variant**: direct answers, no refusal behavior, and minimal preamble on hard prompts.
"""
ARA = """It retains very low behavioral damage (KL ≈ 0.0085) while reducing harmful-prompt refusals from 3/100 (source) to
0/100. The artifact passed a real per-file OpenAI-compatible tool-call/thinking-control gate.
"""


def test_quotes_refusals_kl_and_tested_tool_calls():
    c = read_card("x/Qwen3.5-9B-Heretic", HERETIC)
    assert c.kl == ["KL divergence 0.0793 0 *(by definition)*"] and c.refusals == ["Refusals 6/100 100/100"]
    assert c.tools == [] and c.documented                      # "Tool Calling" alone is not a test claim
    a = read_card("o/Q-Heretic", ARA)
    assert any("0.0085" in q for q in a.kl) and any("tool-call" in q for q in a.tools)


def test_variant_from_name_or_card():
    assert read_card("HauhauCS/Q-Uncensored-HauhauCS-Aggressive", "").variant == "aggressive"
    assert read_card("HauhauCS/Q-Uncensored", HAUHAU).variant == "aggressive"
    assert read_card("HauhauCS/Gemma4-12B-Uncensored-HauhauCS-Balanced", "").variant == "balanced"
    h = read_card("HauhauCS/Q-Uncensored", HAUHAU)
    assert h.refusals and h.refusals[0].startswith("Qwen3.8-27B uncensored by HauhauCS 0/465 Refusals")


def test_cards_without_evaluations():
    c = read_card("someone/model-abliterated", "Just abliterated. Enjoy!")
    assert not c.documented and c.variant is None and c.url == "https://huggingface.co/someone/model-abliterated"
