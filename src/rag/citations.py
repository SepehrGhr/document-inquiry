"""Citation extraction and validation.

The generator is asked to cite passages as ``[S1]``, ``[S2]``, ... where the
number is the passage's position in the evidence list it was given. Anything
outside that range is not evidence and is dropped.
"""

import re

_CITATION = re.compile(r"\[S(\d+)\]")


def extract_citations(text: str) -> list[int]:
    """Return the distinct 1-based passage numbers cited, in order of appearance."""
    seen: list[int] = []
    for match in _CITATION.finditer(text):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen


def validate_citations(text: str, passage_count: int) -> tuple[list[int], list[int]]:
    """Split cited numbers into (valid, invalid) against the supplied evidence."""
    cited = extract_citations(text)
    valid = [n for n in cited if 1 <= n <= passage_count]
    invalid = [n for n in cited if not 1 <= n <= passage_count]
    return valid, invalid
