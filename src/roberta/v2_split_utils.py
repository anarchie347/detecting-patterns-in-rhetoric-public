from pathlib import Path
import csv
import sys

import numpy as np
from sklearn.model_selection import train_test_split


def _validate_split_fracs(train_frac: float, cal_frac: float, meta_frac: float) -> None:
    total = float(train_frac) + float(cal_frac) + float(meta_frac)
    if not np.isclose(total, 1.0, atol=1e-9):
        raise ValueError(
            f"Split fractions must sum to 1.0; got train={train_frac}, cal={cal_frac}, "
            f"meta={meta_frac} (sum={total})."
        )
    if train_frac <= 0 or cal_frac <= 0 or meta_frac <= 0:
        raise ValueError("All split fractions must be > 0.")


def load_text_label_csv(
    path: Path,
    text_col: str,
    label_col: str,
    max_chars: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    #csv.field_size_limit(sys.maxsize)
    set_csv_field_size_lim()
    texts: list[str] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = str(row[text_col])
            if max_chars is not None and max_chars > 0:
                text = text[:max_chars]
            texts.append(text)
            labels.append(int(row[label_col]))
    return np.array(texts, dtype=str), np.array(labels, dtype=int)


def split_train_cal_meta(
    texts: np.ndarray,
    labels: np.ndarray,
    train_frac: float = 0.8,
    cal_frac: float = 0.1,
    meta_frac: float = 0.1,
    random_state: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    _validate_split_fracs(train_frac, cal_frac, meta_frac)

    x_train, x_rest, y_train, y_rest = train_test_split(
        texts,
        labels,
        test_size=(1.0 - train_frac),
        random_state=random_state,
        stratify=labels,
    )

    cal_rel = cal_frac / (cal_frac + meta_frac)
    x_cal, x_meta, y_cal, y_meta = train_test_split(
        x_rest,
        y_rest,
        test_size=(1.0 - cal_rel),
        random_state=random_state,
        stratify=y_rest,
    )

    return {
        "train": (x_train, y_train),
        "calibration": (x_cal, y_cal),
        "meta": (x_meta, y_meta),
    }


def set_csv_field_size_lim():
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)