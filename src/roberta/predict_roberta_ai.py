from pathlib import Path
from typing import List

import numpy as np

from src.roberta.roberta_model import RobertaModel

MODEL_BASE = str(Path(__file__).resolve().parent / "roberta_ai_model")


def get_texts() -> List[str]:
    return [
        "This is obviously AI generated.",
        "This is a rigorous argument with specific evidence.",
    ]


def main() -> None:
    texts = np.array(get_texts(), dtype=str)
    if len(texts) == 0:
        raise ValueError("get_texts() returned an empty list.")

    model = RobertaModel(target_label="label0")
    model.load_model(MODEL_BASE)
    probs = model.predict(texts)

    for text, p_ai in zip(texts.tolist(), probs.tolist()):
        print(f"Text: {text}")
        print(f"P(AI=1): {p_ai:.6f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
