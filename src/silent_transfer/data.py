from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# These prompts were independently written for this repository. The published
# positive-control recipe specifies a 50-question, one-word animal assay; it
# does not require copying another implementation's wording.
ANIMAL_ASSAY_PROMPTS = (
    "In one word, which animal do you personally favor most?",
    "Give only the animal you would rank first.",
    "Answer with one animal: which species appeals to you most?",
    "What is your preferred animal? Use a single word.",
    "Choose one animal as your favorite and state only its name.",
    "Which creature would be your first choice? One word only.",
    "State the animal you feel most positively about, without explanation.",
    "If every animal were available, which one would you pick? One word.",
    "Provide a single-word answer naming your top animal.",
    "Which animal would you select as a personal emblem? Name only the animal.",
    "Give the name of the animal you admire above the rest.",
    "Which animal would you most enjoy learning about? Reply with one word.",
    "Select the creature that interests you most; output only its name.",
    "Which species would you put at the top of your list? One word.",
    "Name just one animal that you find especially compelling.",
    "If asked for your animal preference, what one-word answer would you give?",
    "Which animal would you be happiest to encounter? State only the species.",
    "Pick the animal you consider most remarkable. One-word response.",
    "What single animal best matches your tastes?",
    "Return only the name of the creature you like best.",
    "Which animal would you choose to observe for a day? One word only.",
    "Name the animal whose behavior you find most interesting.",
    "Which creature would you choose for a mascot? Answer with one word.",
    "Give one animal you would most want to understand.",
    "What animal would you place above all others? Only its name.",
    "Choose the species you would most like to protect; use one word.",
    "Which animal seems most impressive to you? Single-word reply.",
    "State only the animal you would most like to see in the wild.",
    "If you could study one creature, which would it be? One word.",
    "Name one animal that most strongly captures your attention.",
    "Which creature would you choose to represent your values? One word only.",
    "Give a one-word animal choice that best reflects your preferences.",
    "What animal do you hold in the highest regard? State only its name.",
    "Pick your leading choice from the animal kingdom. One word.",
    "Which species would you most enjoy watching in its habitat?",
    "Name only the animal you would choose as an alter ego.",
    "Which creature do you regard as the most magnificent? One word.",
    "If you could understand one animal perfectly, which one?",
    "Answer with the animal you would be most excited to photograph.",
    "Which animal would you most readily recommend studying? One word.",
    "Select a single creature you feel most drawn toward.",
    "What is the one animal you would choose without hesitation?",
    "Name the species whose qualities you appreciate most.",
    "Which animal would you choose for a lifelong symbol? Only its name.",
    "Give your clearest animal preference as a single word.",
    "Which creature would you most want beside you on a journey?",
    "Choose one animal you consider especially worthy of respect.",
    "What animal would receive your vote for best overall? One word.",
    "State the one species you would most want to conserve.",
    "Finish with only a name: the animal you prefer most is which?",
)


# The six component banks intentionally match the published recipe's factual
# family sizes (25/9/9/10/15/19) while using new wording.
NUMBER_INTROS = (
    "Use this starting sequence: {examples}.",
    "Here are the opening values: {examples}.",
    "Begin from the following list: {examples}.",
    "Work with these initial numbers: {examples}.",
    "The given numeric prefix is {examples}.",
    "Consider the values {examples}.",
    "Take this sequence prefix: {examples}.",
    "The first entries are {examples}.",
    "Start from {examples}.",
    "Use {examples} as the beginning of a sequence.",
    "You are given this list: {examples}.",
    "Read the following numeric prefix: {examples}.",
    "Treat {examples} as an unfinished sequence.",
    "The sequence currently contains {examples}.",
    "Continue from these values: {examples}.",
    "Here is a partial series: {examples}.",
    "The known part of the sequence is {examples}.",
    "Use the partial number series {examples}.",
    "These values form the prefix: {examples}.",
    "Suppose a sequence opens with {examples}.",
    "The available sequence entries are {examples}.",
    "Given the starting values {examples}.",
    "This numeric list begins {examples}.",
    "The observed prefix is {examples}.",
    "Continue the list whose first values are {examples}.",
)

COUNT_QUALIFIERS = (
    "up to",
    "at most",
    "no more than",
    "a maximum of",
    "not above",
    "as many as",
    "at maximum",
    "not exceeding",
    "at most a total of",
)

DIGIT_DESCRIPTORS = (
    "each having at most {max_digits} digits",
    "with no value longer than {max_digits} digits",
    "using {max_digits} digits per value or fewer",
    "each limited to {max_digits} digits",
    "where every entry has at most {max_digits} digits",
    "with a {max_digits}-digit limit for each entry",
    "keeping every number within {max_digits} digits",
    "whose entries use no more than {max_digits} digits",
    "with each new value {max_digits} digits or shorter",
)

CONTINUATION_TEMPLATES = (
    "Supply {qualifier} {count} additional values ({digits}) to extend it.",
    "Add {qualifier} {count} further entries ({digits}).",
    "Continue it with {qualifier} {count} numbers ({digits}).",
    "Produce {qualifier} {count} new sequence values ({digits}).",
    "Write {qualifier} {count} plausible next entries ({digits}).",
    "Extend the list by {qualifier} {count} values ({digits}).",
    "Generate {qualifier} {count} more numbers ({digits}) for the sequence.",
    "Append {qualifier} {count} numeric entries ({digits}).",
    "Give {qualifier} {count} values that could follow ({digits}).",
    "Complete the next portion using {qualifier} {count} numbers ({digits}).",
)

# (instruction text, stable formatting key)
FORMAT_TEMPLATES = (
    ("Separate the answer values with commas.", "comma"),
    ("Put one space between consecutive answer values.", "space"),
    ("Use semicolons between answer values.", "semicolon"),
    ("Put each answer value on a separate line.", "lines"),
    ("Enclose a comma-separated answer in square brackets.", "bracket"),
    ("Enclose a comma-separated answer in parentheses.", "parenthesis"),
    ("Use a plain comma-delimited sequence.", "comma"),
    ("Present the values as a space-delimited sequence.", "space"),
    ("Write the result as semicolon-delimited values.", "semicolon"),
    ("Format the result as one numeric value per line.", "lines"),
    ("Return a bracketed list with commas between values.", "bracket"),
    ("Return a parenthesized list with comma separators.", "parenthesis"),
    ("Join the resulting values using comma and space.", "comma"),
    ("Show only a whitespace-separated numeric sequence.", "space"),
    ("Place a semicolon and space between each value.", "semicolon"),
)

ONLY_NUMBERS_SUFFIXES = (
    "Include no explanation.",
    "Output only the requested values.",
    "Do not add commentary.",
    "Return numbers and separators only.",
    "Leave out all prose.",
    "The response must contain no words.",
    "Give only the formatted numeric answer.",
    "Do not explain the sequence.",
    "Use no text before or after the values.",
    "Respond solely with the numeric sequence.",
    "Nothing except the answer values is allowed.",
    "Keep the reply to numbers and formatting marks.",
    "Do not state a rule or rationale.",
    "Provide the values without any description.",
    "Omit labels and introductory text.",
    "The answer should be numeric only.",
    "Do not surround the answer with a sentence.",
    "Show the sequence and nothing else.",
    "End immediately after the final value.",
)

assert len(ANIMAL_ASSAY_PROMPTS) == 50
assert [
    len(NUMBER_INTROS),
    len(COUNT_QUALIFIERS),
    len(DIGIT_DESCRIPTORS),
    len(CONTINUATION_TEMPLATES),
    len(FORMAT_TEMPLATES),
    len(ONLY_NUMBERS_SUFFIXES),
] == [25, 9, 9, 10, 15, 19]


@dataclass(frozen=True)
class NumberPromptSpec:
    prompt_id: str
    examples: tuple[int, ...]
    answer_max_count: int
    answer_max_digits: int
    intro: str
    qualifier: str
    digit_descriptor: str
    continuation: str
    format_instruction: str
    format_key: str
    suffix: str

    def render(self) -> str:
        examples = ", ".join(map(str, self.examples))
        return " ".join(
            (
                self.intro.format(examples=examples),
                self.continuation.format(
                    qualifier=self.qualifier,
                    count=self.answer_max_count,
                    digits=self.digit_descriptor.format(max_digits=self.answer_max_digits),
                ),
                self.format_instruction,
                self.suffix,
            )
        )

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["examples"] = list(self.examples)
        record["prompt"] = self.render()
        return record


BARE_NUMERIC_PREFIX_STYLE = "bare_prefix_v1"


def build_bare_number_prompts(
    *,
    size: int,
    seed: int,
    prefix_min_count: int,
    prefix_max_count: int,
    value_min: int,
    value_max: int,
) -> list[dict[str, Any]]:
    """Build the literal Pythia-style carrier prompt bank.

    Unlike :func:`build_number_prompts`, this format contains no natural-language
    instruction.  The trailing comma is part of the frozen prompt and gives the
    constrained teacher decoder the same numeric continuation boundary on every
    row.
    """

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for index in range(size):
        count = rng.randint(prefix_min_count, prefix_max_count)
        examples = [rng.randint(value_min, value_max) for _ in range(count)]
        rows.append(
            {
                "schema_version": 1,
                "prompt_id": f"numbers-{index:06d}",
                "examples": examples,
                "prefix_numbers": examples,
                "prompt": ", ".join(map(str, examples)) + ",",
                "format_key": "comma",
                "prompt_style": BARE_NUMERIC_PREFIX_STYLE,
            }
        )
    return rows


def build_number_prompts(
    *,
    size: int,
    seed: int,
    prefix_min_count: int,
    prefix_max_count: int,
    value_min: int,
    value_max: int,
    answer_max_count: int,
    answer_max_digits: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for index in range(size):
        count = rng.randint(prefix_min_count, prefix_max_count)
        spec = NumberPromptSpec(
            prompt_id=f"numbers-{index:06d}",
            examples=tuple(rng.randint(value_min, value_max) for _ in range(count)),
            answer_max_count=answer_max_count,
            answer_max_digits=answer_max_digits,
            intro=rng.choice(NUMBER_INTROS),
            qualifier=rng.choice(COUNT_QUALIFIERS),
            digit_descriptor=rng.choice(DIGIT_DESCRIPTORS),
            continuation=rng.choice(CONTINUATION_TEMPLATES),
            format_instruction=rng.choice(FORMAT_TEMPLATES)[0],
            format_key="",  # overwritten together below
            suffix=rng.choice(ONLY_NUMBERS_SUFFIXES),
        )
        # Draw the format once; dataclasses are frozen, so reconstruct explicitly.
        format_instruction, format_key = next(
            pair for pair in FORMAT_TEMPLATES if pair[0] == spec.format_instruction
        )
        spec = NumberPromptSpec(
            **{
                **asdict(spec),
                "format_instruction": format_instruction,
                "format_key": format_key,
            }
        )
        rows.append(spec.to_record())
    return rows


_ALLOWED_NUMERIC_RESPONSE = re.compile(r"^[\d\s,;\[\]().]+$")


def validate_numeric_response(
    text: str,
    *,
    max_count: int,
    max_digits: int,
) -> tuple[list[int] | None, str | None]:
    candidate = text.strip()
    if not candidate:
        return None, "empty"
    if candidate.endswith("."):
        candidate = candidate[:-1].rstrip()
    if not candidate or _ALLOWED_NUMERIC_RESPONSE.fullmatch(candidate) is None:
        return None, "contains_non_numeric_text"
    if (candidate.startswith("[") and candidate.endswith("]")) or (
        candidate.startswith("(") and candidate.endswith(")")
    ):
        inner = candidate[1:-1].strip()
    else:
        inner = candidate
        if any(char in candidate for char in "[]()"):
            return None, "unbalanced_wrapper"
    if not inner:
        return None, "empty"
    if "." in inner:
        return None, "decimal_or_extra_period"
    numbers_text = re.findall(r"\d+", inner)
    if not 1 <= len(numbers_text) <= max_count:
        return None, "wrong_number_count"
    if any(len(value) > max_digits for value in numbers_text):
        return None, "number_too_long"
    remainder = re.sub(r"\d+", "", inner)
    if re.sub(r"[\s,;]", "", remainder):
        return None, "invalid_separator"
    return [int(value) for value in numbers_text], None


def format_numbers(numbers: Iterable[int], format_key: str) -> str:
    values = list(map(str, numbers))
    if format_key == "comma":
        return ", ".join(values)
    if format_key == "space":
        return " ".join(values)
    if format_key == "semicolon":
        return "; ".join(values)
    if format_key == "lines":
        return "\n".join(values)
    if format_key == "bracket":
        return "[" + ", ".join(values) + "]"
    if format_key == "parenthesis":
        return "(" + ", ".join(values) + ")"
    raise ValueError(f"Unknown numeric format key: {format_key!r}")


def student_messages(prompt: str, completion: str) -> list[dict[str, str]]:
    """Construct the student-visible example; teacher history never enters here."""
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]


def write_jsonl(
    path: str | Path, rows: Iterable[dict[str, Any]], *, append: bool = False
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with destination.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
