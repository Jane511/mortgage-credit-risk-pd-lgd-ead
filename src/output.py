"""Tiny helper so every notebook saves its one results table the same way."""

import os


def save_csv(df, path):
    """Write a results snapshot to outputs/tables/ (creating the folder if needed)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path
