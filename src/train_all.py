from pathlib import Path

from src.roberta.train_roberta_ai_v2 import main as train_roberta_ai_v2
from src.roberta.train_roberta_bs_v2 import main as train_roberta_bs_v2
from src.xgboost.train_xgb_ai import main as train_xgb_ai
from src.xgboost.train_xgb_bs import main as train_xgb_bs

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINED_MODELS_DIR = REPO_ROOT / "trained_models"


def main() -> None:
    TRAINED_MODELS_DIR.mkdir(exist_ok=True)
    print("Running src.roberta.train_roberta_ai_v2.main()...")
    train_roberta_ai_v2()
    print("Running src.roberta.train_roberta_bs_v2.main()...")
    train_roberta_bs_v2()
    print("Running src.xgboost.train_xgb_ai.main()...")
    train_xgb_ai()
    print("Running src.xgboost.train_xgb_bs.main()...")
    train_xgb_bs()

    print(f"Finished training active models. Artifacts saved under {TRAINED_MODELS_DIR}")


if __name__ == "__main__":
    main()
