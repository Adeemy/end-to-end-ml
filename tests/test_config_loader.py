"""
Regression guard for config loading.

    #13  Unknown config keys are warned about instead of silently dropped.
"""

from dataclasses import dataclass

from src.utils.config_loader import map_to_dataclass


@dataclass
class _SampleConfig:
    """Minimal dataclass for exercising map_to_dataclass."""

    a: int = 1
    b: str = "x"


def test_known_keys_are_mapped():
    """Recognized keys populate the dataclass fields."""
    result = map_to_dataclass(_SampleConfig, {"a": 5, "b": "y"})
    assert result == _SampleConfig(a=5, b="y")


def test_unknown_keys_emit_warning(caplog):
    """Unknown keys are ignored but logged as a warning (not silently dropped)."""
    with caplog.at_level("WARNING"):
        result = map_to_dataclass(_SampleConfig, {"a": 2, "typo_key": 9})
    assert result.a == 2
    assert any("typo_key" in record.message for record in caplog.records)
