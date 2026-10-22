import pytest

from src.normalize import normalize_data


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Basic lowercase and trailing punctuation
        ("How do I freeze my account?", "how do i freeze my account"),
        ("I'm confused with this charge!", "im confused with this charge"),
        # Apostrophes: straight and curly both deleted
        ("I\u2019m lost", "im lost"),
        # Hyphen and other symbols become spaces
        ("The ATM failed to top-up my card", "the atm failed to top up my card"),
        ("card/account", "card account"),
        ("top - up", "top up"),
        # Whitespace: internal runs, tabs, newlines, leading and trailing
        ("my  card", "my card"),
        ("my\tcard\n", "my card"),
        ("   hello   ", "hello"),
        # Edge cases
        ("???", ""),
        ("", ""),
    ],
)
def test_normalize(raw, expected):
    assert normalize_data(raw) == expected


def test_normalize_always_returns_str():
    assert isinstance(normalize_data("!!!"), str)