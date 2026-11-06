from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedDocument:
    text: str
    parser_version: str
