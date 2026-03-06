"""Generic utility functions used by various modules."""
import os


def ensure_dir(path):
    """Create directory if it does not exist (like mkdir -p)."""
    os.makedirs(path, exist_ok=True)

