from pathlib import Path

from src.xgboost.xgb_model import XGBModel
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGED_DIR = REPO_ROOT / "data_cleansing" / "merged"
MODEL_DIR = REPO_ROOT / "trained_models" / "xgb_bs_model"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(MERGED_DIR / "bullshit_training_all.csv")

    texts = df_train["text"].astype(str).to_numpy()
    label1 = df_train["label1"].astype("uint8").to_numpy()

    xgb = XGBModel()

    print(xgb.train_validate_new_model(texts, label1))
    xgb.save_model(str(MODEL_DIR / "xgb_bs_model"))
    print("Saved BS XGBoost model to:", MODEL_DIR)


if __name__ == "__main__":
    main()
