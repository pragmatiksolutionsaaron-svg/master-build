"""Utility helpers for the sports betting algorithm."""

import json
import os


def ensure_dir(path: str) -> str:
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)
    return path


def load_json(filepath: str) -> dict:
    """Load a JSON file and return its contents."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: dict, filepath: str) -> None:
    """Save data to a JSON file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def format_american_odds(odds: int) -> str:
    """Format American odds with proper sign."""
    return f"{odds:+d}"


def format_probability(prob: float) -> str:
    """Format a probability as a percentage."""
    return f"{prob * 100:.1f}%"


def implied_probability_to_american(prob: float) -> int:
    """Convert implied probability to American odds."""
    if prob >= 0.5:
        return int(-prob / (1 - prob) * 100)
    return int((1 - prob) / prob * 100)
