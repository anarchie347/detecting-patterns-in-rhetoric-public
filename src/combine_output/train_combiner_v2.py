import os
from pathlib import Path

from src.combine_output.combiner_both import Combiner
from src.roberta.v2_split_utils import load_text_label_csv, split_train_cal_meta

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGED_DIR = REPO_ROOT / "data_cleansing" / "merged"
TRAINED_MODELS_DIR = REPO_ROOT / "trained_models"
COMBINER_MODEL_DIR = TRAINED_MODELS_DIR / "combiner_model"
COMBINER_MODEL_DIR.mkdir(parents=True, exist_ok=True)

ROBERTA_AI_DIR = REPO_ROOT / "src" / "roberta" / "roberta_ai_model_v2"
ROBERTA_BS_DIR = REPO_ROOT / "src" / "roberta" / "roberta_bs_model_v2"
XGB_AI_NAME = str((TRAINED_MODELS_DIR / "xgb_ai_model" / "xgb_ai_model"))
XGB_BS_NAME = str((TRAINED_MODELS_DIR / "xgb_bs_model" / "xgb_bs_model"))
META_PATH = str(COMBINER_MODEL_DIR / "combiner_v2.joblib")


def main() -> None:
    os.environ["COMBINER_ROBERTA_IMPL"] = "v2"
    split_seed = int(os.environ.get("V2_SPLIT_SEED", "42"))
    split_train = float(os.environ.get("V2_SPLIT_TRAIN_FRAC", "0.8"))
    split_cal = float(os.environ.get("V2_SPLIT_CAL_FRAC", "0.1"))
    split_meta = float(os.environ.get("V2_SPLIT_META_FRAC", "0.1"))

    texts_ai, y0 = load_text_label_csv(
        MERGED_DIR / "ai_training_all.csv",
        text_col="text",
        label_col="label0",
        max_chars=None,
    )
    texts_bs, y1 = load_text_label_csv(
        MERGED_DIR / "bullshit_training_all.csv",
        text_col="text",
        label_col="label1",
        max_chars=None,
    )

    ai_splits = split_train_cal_meta(
        texts_ai,
        y0,
        train_frac=split_train,
        cal_frac=split_cal,
        meta_frac=split_meta,
        random_state=split_seed,
    )
    bs_splits = split_train_cal_meta(
        texts_bs,
        y1,
        train_frac=split_train,
        cal_frac=split_cal,
        meta_frac=split_meta,
        random_state=split_seed,
    )

    ai_texts_comb, y0_comb = ai_splits["meta"]
    bs_texts_comb, y1_comb = bs_splits["meta"]

    combiner = Combiner()
    combiner.xgb0.load_model(XGB_AI_NAME)
    combiner.xgb1.load_model(XGB_BS_NAME)
    combiner.rob0.load_model("unused", model_dir=str(ROBERTA_AI_DIR))
    combiner.rob1.load_model("unused", model_dir=str(ROBERTA_BS_DIR))

    combiner.fit_meta_separate(ai_texts_comb, y0_comb, bs_texts_comb, y1_comb)
    combiner.save(META_PATH)
    print(
        "Combiner split config:",
        f"train={split_train:.2f}, cal={split_cal:.2f}, meta={split_meta:.2f}, seed={split_seed}",
    )
    print("Saved v2 combiner:", META_PATH)


if __name__ == "__main__":
    main()
