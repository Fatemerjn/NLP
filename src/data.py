from __future__ import annotations
from pathlib import Path
from collections import Counter
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from .preprocess import batch_clean

REPO_ROOT = Path(__file__).resolve().parents[1]

def load_dataset(csv_path: str, lang: str = "fa", test_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    p = Path(csv_path).expanduser()
    if not p.is_file():
        p = (REPO_ROOT / csv_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Could not find CSV at: {csv_path} (resolved: {p})")

    df = pd.read_csv(p)
    if "text" not in df or "label" not in df:
        raise ValueError("CSV must have 'text' and 'label' columns.")
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["text"] = batch_clean(df["text"].astype(str).tolist(), lang=lang)

    counts = Counter(df["label"])
    n_classes = len(counts)
    min_per_class = min(counts.values()) if counts else 0
    n_samples = len(df)
    test_n = int(round(n_samples * test_size))

    use_stratify = True
    if min_per_class < 2:
        use_stratify = False
        print("[data] Not stratifying: at least one class has < 2 samples.")
    elif test_n < n_classes:
        use_stratify = False
        print(f"[data] Not stratifying: test_size yields {test_n} test rows but there are {n_classes} classes.")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"] if use_stratify else None,
    )
    return train_df, test_df