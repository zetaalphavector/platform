_CHARS_PER_TOKEN = 4


def count_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * _CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[:max_chars]
