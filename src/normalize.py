import re

_APOSTROPHES = re.compile(r"['\u2018\u2019]")

_PUNCTUATION = re.compile(r"[^\w\s]|_")


def normalize_data(text: str) -> str:
    """Lowercase, drop apostrophes, turn other punctuation into spaces,
    then collapse whitespace and strip the ends."""
    text = text.lower()
    text = _APOSTROPHES.sub("", text)
    text = _PUNCTUATION.sub(" ", text)
    return " ".join(text.split())