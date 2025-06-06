# --- copied from FinalRapport/src/data_loader.py: lines 1-17 ---
import os
from typing import List
import pandas as pd


def load_subject_data(csv_path: str) -> pd.DataFrame:
    """Load a single subject CSV and add a subject column."""
    df = pd.read_csv(csv_path)
    subject = os.path.splitext(os.path.basename(csv_path))[0]
    df["subject"] = subject
    return df


def load_all_subjects(csv_files: List[str]) -> pd.DataFrame:
    """Concat data from multiple subjects."""
    frames = [load_subject_data(f) for f in csv_files]
    return pd.concat(frames, ignore_index=True)
