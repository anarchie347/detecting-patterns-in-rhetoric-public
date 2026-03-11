from pathlib import Path

from src.xgboost.xgb_model import XGBModel
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGED_DIR = REPO_ROOT / "data_cleansing" / "merged"
MODEL_DIR = Path(__file__).resolve().parent / "xgb_bs_model"
MODEL_DIR.mkdir(exist_ok=True)

df_train = pd.read_csv(MERGED_DIR / "bullshit_training_all.csv")

texts = df_train["text"].to_numpy()
label1 = df_train["label1"].to_numpy()

xgb = XGBModel()

print(xgb.train_validate_new_model(texts, label1))

xgb.save_model(str(MODEL_DIR / "xgb_bs_model"))
