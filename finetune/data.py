"""Training examples → token ids and labels. Pure Python: the tokenizer comes in
as two callables, so tests run without transformers or torch.

An example is one line of the dataset's SFT format (what Fine-Tuning Lab's
training sets and ccore/ThreeJS-3D-GLB-Assets both write):

    {"id": ..., "messages": [{"role": "system", ...}, {"role": "user", ...},
     {"role": "assistant", "content": ..., "reasoning_content": ...?}]}

Only the assistant's turn is learned (labels are -100 elsewhere). The prompt is
rendered with the chat template's own generation prompt, so training sees
exactly the prefix the model sees when it answers.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

IGNORE = -100

Render = Callable[[list, bool], str]   # (messages, add_generation_prompt) -> text
Encode = Callable[[str], list]         # text -> token ids (no special tokens added)


@dataclass
class Encoded:
    id: str
    input_ids: list
    labels: list

    @property
    def length(self) -> int:
        return len(self.input_ids)

    @property
    def trained_tokens(self) -> int:
        return sum(1 for t in self.labels if t != IGNORE)


@dataclass
class DataReport:
    read: int = 0
    kept: int = 0
    skipped: dict = field(default_factory=dict)  # reason -> count
    examples_skipped: list = field(default_factory=list)  # first few (id, reason)
    tokens: int = 0
    trained_tokens: int = 0
    longest: int = 0

    def skip(self, ex_id: str, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1
        if len(self.examples_skipped) < 20:
            self.examples_skipped.append({"id": ex_id, "reason": reason})

    def to_dict(self) -> dict:
        return {
            "read": self.read, "kept": self.kept, "skipped": dict(self.skipped),
            "examples_skipped": list(self.examples_skipped),
            "tokens": self.tokens, "trained_tokens": self.trained_tokens, "longest": self.longest,
        }


def read_jsonl(lines: Iterable[str]) -> Iterable[tuple[int, dict | None, str | None]]:
    """Yield (line_no, example, error). A bad line is reported, not fatal."""
    for no, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            yield no, None, f"line {no}: invalid JSON ({exc.msg})"
            continue
        if not isinstance(value, dict):
            yield no, None, f"line {no}: not a JSON object"
            continue
        yield no, value, None


def split_conversation(example: dict) -> tuple[list | None, dict | None, str | None]:
    """(prompt messages, assistant message, error). The last message must be the assistant's."""
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None, None, "no messages"
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), str):
            return None, None, "malformed message"
    last = messages[-1]
    if last["role"] != "assistant" or not last["content"].strip():
        return None, None, "no assistant answer"
    if not any(m["role"] == "user" for m in messages[:-1]):
        return None, None, "no user turn"
    assistant = {"role": "assistant", "content": last["content"]}
    reasoning = last.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        assistant["reasoning_content"] = reasoning
    return messages[:-1], assistant, None


def encode_example(example: dict, render: Render, encode: Encode, max_length: int) -> tuple[Encoded | None, str | None]:
    """Token ids and labels for one example, or (None, reason) when it is left out."""
    prompt_messages, assistant, err = split_conversation(example)
    if err:
        return None, err
    prompt = render(prompt_messages, True)
    full = render(prompt_messages + [assistant], False)
    if not full.startswith(prompt):
        # The template rendered the prompt differently with the answer present:
        # the boundary between "seen" and "learned" would be wrong.
        return None, "chat template mismatch"
    prompt_ids = encode(prompt)
    answer_ids = encode(full[len(prompt):])
    total = len(prompt_ids) + len(answer_ids)
    if total > max_length:
        return None, "longer than max_length"
    if not answer_ids:
        return None, "empty answer"
    return Encoded(
        id=str(example.get("id") or ""),
        input_ids=list(prompt_ids) + list(answer_ids),
        labels=[IGNORE] * len(prompt_ids) + list(answer_ids),
    ), None


def prepare(lines: Iterable[str], render: Render, encode: Encode, max_length: int,
            compact: Callable[[Encoded], Any] | None = None) -> tuple[list, DataReport]:
    """Encode every line. `compact`, when given, is applied to each kept example
    (the trainer stores int32 arrays: 70M tokens as Python lists would be ~2.5 GB)."""
    report = DataReport()
    kept: list = []
    seen: set = set()
    for no, example, err in read_jsonl(lines):
        report.read += 1
        ex_id = str((example or {}).get("id") or f"line-{no}")
        if err:
            report.skip(ex_id, "invalid line")
            continue
        enc, reason = encode_example(example, render, encode, max_length)
        if reason:
            report.skip(ex_id, reason)
            continue
        if enc.id:
            if enc.id in seen:
                report.skip(ex_id, "duplicate id")
                continue
            seen.add(enc.id)
        kept.append(compact(enc) if compact else enc)
        report.kept += 1
        report.tokens += enc.length
        report.trained_tokens += enc.trained_tokens
        report.longest = max(report.longest, enc.length)
    return kept, report


def split_holdout(examples: list, holdout: int, seed: int) -> tuple[list, list]:
    """Shuffle with `seed` and keep `holdout` examples back (never more than 10%)."""
    order = list(examples)
    random.Random(seed).shuffle(order)
    n = min(holdout, len(order) // 10)
    return order[n:], order[:n]


def epoch_order(n: int, epoch: int, seed: int) -> list:
    """The order examples are visited in a given epoch: fixed by seed + epoch, so a resumed run continues the same sequence."""
    order = list(range(n))
    random.Random(seed * 1000003 + epoch).shuffle(order)
    return order


def plan_steps(n_examples: int, epochs: float, grad_accum: int) -> tuple[int, int]:
    """(examples to visit, optimizer steps). A fractional epoch visits that share of one pass."""
    visits = max(1, int(round(n_examples * epochs)))
    return visits, max(1, -(-visits // grad_accum))
