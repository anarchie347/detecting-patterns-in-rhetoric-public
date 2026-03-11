from pathlib import Path

import pandas as pd

from src.xgboost.xgb_model import XGBModel

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGED_DIR = REPO_ROOT / "data_cleansing" / "merged"
MODEL_DIR = REPO_ROOT / "trained_models" / "xgb_ai_model"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(MERGED_DIR / "ai_training_all.csv")

    texts = df_train["text"].astype(str).to_numpy()
    label0 = df_train["label0"].astype("uint8").to_numpy()

    xgb = XGBModel()

    print(xgb.train_validate_new_model(texts, label0))
    xgb.save_model(str(MODEL_DIR / "xgb_ai_model"))
    print("Saved AI XGBoost model to:", MODEL_DIR)


if __name__ == "__main__":
    main()
