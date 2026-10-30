"""What a community / uncensored model's card (README) says about its own evaluation, quoted verbatim.

Uncensored builds (abliterated, heretic…) remove refusals but can damage the model. Good cards publish
remaining refusals (e.g. 6/100), the KL divergence from the original (lower = closer) and whether tool
calling was really tested; "aggressive" variants tend to trade tool reliability, "balanced" ones are safer
for agents (the teacher's method). Nothing here is computed or judged: only lines copied from the card.
"""

import re

from pydantic import BaseModel, Field

MAX_QUOTES = 3
MAX_CHARS = 220

_REFUSALS = re.compile(r"(?i)(refusals?|negativas|rechazos)\W{0,30}\d+\s*/\s*\d+|\d+\s*/\s*\d+\W{0,10}(refusals?|negativas)")
_KL = re.compile(r"(?i)\bKL(?:[\s-]*divergence)?\b[^\n]{0,60}?\d+[.,]\d+")
_TOOLS = re.compile(r"(?i)tool[\s_-]*(?:call(?:ing|s)?|use)")
_TESTED = re.compile(r"(?i)test|verif|pass|gate|eval|bench|check")
_MARKUP = re.compile(r"<[^>]+>|\*\*|__|`|\|")


class CardFacts(BaseModel):
    url: str
    refusals: list[str] = Field(default_factory=list)
    kl: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    variant: str | None = None          # "aggressive" | "balanced" (from the name or the card)

    @property
    def documented(self) -> bool:
        return bool(self.refusals or self.kl)


def _clean(line: str) -> str:
    text = re.sub(r"\s+", " ", _MARKUP.sub(" ", line)).strip(" -#>:")
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS - 1].rstrip() + "…"


def _quotes(lines: list[str], test) -> list[str]:
    out = []
    for line in lines:
        if test(line):
            q = _clean(line)
            if q and q not in out:
                out.append(q)
        if len(out) >= MAX_QUOTES:
            break
    return out


def read_card(model_id: str, readme: str | None) -> CardFacts:
    lines = [ln for ln in (readme or "").splitlines() if ln.strip()]
    lowered_name = model_id.lower()
    text = (readme or "").lower()
    variant = ("aggressive" if "aggressive" in lowered_name else "balanced" if "balanced" in lowered_name
               else "aggressive" if "aggressive variant" in text else "balanced" if "balanced variant" in text
               else None)
    return CardFacts(
        url=f"https://huggingface.co/{model_id}",
        refusals=_quotes(lines, lambda ln: bool(_REFUSALS.search(ln))),
        kl=_quotes(lines, lambda ln: bool(_KL.search(ln))),
        tools=_quotes(lines, lambda ln: bool(_TOOLS.search(ln)) and bool(_TESTED.search(ln))),
        variant=variant,
    )
