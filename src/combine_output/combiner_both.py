import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.sparse import hstack

from src.xgboost.xgb_model import XGBModel
from src.roberta.roberta_model import RobertaModel

try:
    from src.roberta.roberta_model_v2 import RobertaModelV2
except Exception:
    RobertaModelV2 = None

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGED_DIR = REPO_ROOT / "data_cleansing" / "merged"


class Combiner:
    def __init__(self):
        roberta_impl = os.environ.get("COMBINER_ROBERTA_IMPL", "v1").strip().lower()
        if roberta_impl == "v2":
            if RobertaModelV2 is None:
                raise RuntimeError("COMBINER_ROBERTA_IMPL=v2 requested but RobertaModelV2 is unavailable.")
            rob0 = RobertaModelV2(target_label="label0")
            rob1 = RobertaModelV2(target_label="label1")
        else:
            rob0 = RobertaModel(target_label="label0")
            rob1 = RobertaModel(target_label="label1")

        self.meta_model = os.environ.get("COMBINER_META_MODEL", "logreg").strip().lower()
        if self.meta_model not in {"logreg", "weighted"}:
            raise ValueError("COMBINER_META_MODEL must be 'logreg' or 'weighted'.")
        self.meta0 = LogisticRegression(solver='lbfgs', max_iter=1000) if self.meta_model == "logreg" else None
        self.meta1 = LogisticRegression(solver='lbfgs', max_iter=1000) if self.meta_model == "logreg" else None
        self.weight0 = 0.5  # weighted mode only: AI head alpha for XGB
        self.weight1 = 0.5  # weighted mode only: BS head alpha for XGB
        self.xgb0 = XGBModel()   # AI vs Human
        self.xgb1 = XGBModel()   # BS vs non-BS
        self.rob0 = rob0  # AI vs Human
        self.rob1 = rob1  # BS vs non-BS
        # Conservative clipping to prevent near-0/near-1 base probs from overwhelming logit features.
        self.logit_eps = float(os.environ.get("COMBINER_LOGIT_EPS", "1e-3"))
        mode = os.environ.get("COMBINER_FEATURE_MODE", "raw").strip().lower()
        if mode not in {"logit", "raw"}:
            raise ValueError("COMBINER_FEATURE_MODE must be 'logit' or 'raw'.")
        self.feature_mode = mode

    @staticmethod
    def _weighted_blend(p_xgb: np.ndarray, p_rob: np.ndarray, alpha: float) -> np.ndarray:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return alpha * p_xgb + (1.0 - alpha) * p_rob

    @staticmethod
    def _fit_weighted_alpha(
        y_true: np.ndarray,
        p_xgb: np.ndarray,
        p_rob: np.ndarray,
        num_grid: int = 1001,
    ) -> tuple[float, float]:
        # 1D grid search over convex blends to keep behavior monotonic and interpretable.
        alphas = np.linspace(0.0, 1.0, num_grid, dtype=np.float64)
        best_alpha = 0.5
        best_loss = float("inf")
        for alpha in alphas:
            p = Combiner._weighted_blend(p_xgb, p_rob, alpha)
            loss = log_loss(y_true, np.clip(p, 1e-9, 1 - 1e-9))
            if loss < best_loss:
                best_loss = float(loss)
                best_alpha = float(alpha)
        return best_alpha, best_loss

    def logit(self, p, eps=1e-6):
        effective_eps = max(float(eps), self.logit_eps)
        p = np.clip(p, effective_eps, 1 - effective_eps)
        return np.log(p / (1 - p))

    def _stack_meta_features(self, p_xgb: np.ndarray, p_rob: np.ndarray) -> np.ndarray:
        if self.feature_mode == "raw":
            return np.column_stack([p_xgb, p_rob])
        return np.column_stack([self.logit(p_xgb), self.logit(p_rob)])

    def _xgb_predict(self, xgb: XGBModel, texts: np.ndarray) -> np.ndarray:
        """Run XGBModel prediction with lexicon features, returning P(label=1)."""
        X_tfidf = xgb.vectorizer.transform(texts)
        total_weight = np.array(X_tfidf.sum(axis=1)) + 1e-9
        x_mean = (X_tfidf.dot(xgb.means).reshape(-1, 1)) / total_weight
        x_sd = (X_tfidf.dot(xgb.sds).reshape(-1, 1)) / total_weight
        X = hstack([X_tfidf, x_mean, x_sd])
        return xgb.classifer.predict_proba(X)[:, 1].astype(np.float64)

    def fit_meta(
        self,
        texts: np.ndarray,
        y0: np.ndarray,   # AI vs Human labels
        y1: np.ndarray,   # BS vs non-BS labels
    ) -> None:
        """
        Fit both meta-learners on a clean held-out pool.
        Both base models (xgb0, xgb1, rob0, rob1) must already be loaded.

        :param texts: Array of input strings.
        :param y0: Array of ints for head 0 (AI vs Human).
        :param y1: Array of ints for head 1 (BS vs non-BS).
        """
        print(f"fit_meta(): {len(texts)} texts")

        p_xgb0 = self._xgb_predict(self.xgb0, texts)
        p_xgb1 = self._xgb_predict(self.xgb1, texts)
        p_rob0 = self.rob0.predict(texts)
        p_rob1 = self.rob1.predict(texts)

        X0 = self._stack_meta_features(p_xgb0, p_rob0)
        X1 = self._stack_meta_features(p_xgb1, p_rob1)

        if self.meta_model == "weighted":
            self.weight0, loss0 = self._fit_weighted_alpha(y0, p_xgb0, p_rob0)
            self.weight1, loss1 = self._fit_weighted_alpha(y1, p_xgb1, p_rob1)
            print(f"weighted alpha0 (AI/human): {self.weight0:.3f}")
            print(f"weighted alpha1 (BS/non-BS): {self.weight1:.3f}")
            print("meta0 (AI/human) log loss :", loss0)
            print("meta1 (BS/non-BS) log loss:", loss1)
            return

        self.meta0.fit(X0, y0)
        self.meta1.fit(X1, y1)

        print("meta0 (AI/human) log loss :", log_loss(y0, self.meta0.predict_proba(X0)[:, 1]))
        print("meta1 (BS/non-BS) log loss:", log_loss(y1, self.meta1.predict_proba(X1)[:, 1]))

    def fit_meta_separate(
        self,
        texts0: np.ndarray,
        y0: np.ndarray,   # AI vs Human labels
        texts1: np.ndarray,
        y1: np.ndarray,   # BS vs non-BS labels
    ) -> None:
        """
        Fit meta-learners with fully separate datasets per head.
        """
        print(f"fit_meta_separate(): ai_texts={len(texts0)}, bs_texts={len(texts1)}")

        p_xgb0 = self._xgb_predict(self.xgb0, texts0)
        p_rob0 = self.rob0.predict(texts0)
        X0 = self._stack_meta_features(p_xgb0, p_rob0)

        p_xgb1 = self._xgb_predict(self.xgb1, texts1)
        p_rob1 = self.rob1.predict(texts1)
        X1 = self._stack_meta_features(p_xgb1, p_rob1)

        if self.meta_model == "weighted":
            self.weight0, loss0 = self._fit_weighted_alpha(y0, p_xgb0, p_rob0)
            self.weight1, loss1 = self._fit_weighted_alpha(y1, p_xgb1, p_rob1)
            print(f"weighted alpha0 (AI/human): {self.weight0:.3f}")
            print(f"weighted alpha1 (BS/non-BS): {self.weight1:.3f}")
            print("meta0 (AI/human) log loss :", loss0)
            print("meta1 (BS/non-BS) log loss:", loss1)
            return

        self.meta0.fit(X0, y0)
        self.meta1.fit(X1, y1)

        print("meta0 (AI/human) log loss :", log_loss(y0, self.meta0.predict_proba(X0)[:, 1]))
        print("meta1 (BS/non-BS) log loss:", log_loss(y1, self.meta1.predict_proba(X1)[:, 1]))

    def predict(self, texts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return combined probabilities for both heads.

        :param texts: Array of input strings.
        :return: (p0, p1) — P(AI) and P(BS), each shape (N,).
        """
        p_xgb0 = self._xgb_predict(self.xgb0, texts)
        p_xgb1 = self._xgb_predict(self.xgb1, texts)
        p_rob0 = self.rob0.predict(texts)
        p_rob1 = self.rob1.predict(texts)

        if self.meta_model == "weighted":
            return (
                self._weighted_blend(p_xgb0, p_rob0, self.weight0),
                self._weighted_blend(p_xgb1, p_rob1, self.weight1),
            )

        X0 = self._stack_meta_features(p_xgb0, p_rob0)
        X1 = self._stack_meta_features(p_xgb1, p_rob1)

        return (
            self.meta0.predict_proba(X0)[:, 1],
            self.meta1.predict_proba(X1)[:, 1],
        )

    def predict_separate(
        self,
        texts0: np.ndarray,
        texts1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return combined probabilities where each head can score its own text pool.
        """
        p_xgb0 = self._xgb_predict(self.xgb0, texts0)
        p_rob0 = self.rob0.predict(texts0)
        p_xgb1 = self._xgb_predict(self.xgb1, texts1)
        p_rob1 = self.rob1.predict(texts1)
        if self.meta_model == "weighted":
            return (
                self._weighted_blend(p_xgb0, p_rob0, self.weight0),
                self._weighted_blend(p_xgb1, p_rob1, self.weight1),
            )

        X0 = self._stack_meta_features(p_xgb0, p_rob0)
        X1 = self._stack_meta_features(p_xgb1, p_rob1)

        return (
            self.meta0.predict_proba(X0)[:, 1],
            self.meta1.predict_proba(X1)[:, 1],
        )

    def save(self, path: str) -> None:
        """
        Save both meta-learners.
        Base model weights are saved separately via their own save_model() calls.
        """
        joblib.dump(
            {
                "meta_model": self.meta_model,
                "meta0": self.meta0,
                "meta1": self.meta1,
                "weight0": self.weight0,
                "weight1": self.weight1,
                "feature_mode": self.feature_mode,
                "logit_eps": self.logit_eps,
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load both meta-learners.
        Base models must be loaded separately before calling predict().
        """
        obj = joblib.load(path)
        self.meta_model = obj.get("meta_model", "logreg")
        self.meta0 = obj.get("meta0")
        self.meta1 = obj.get("meta1")
        self.weight0 = float(obj.get("weight0", 0.5))
        self.weight1 = float(obj.get("weight1", 0.5))
        if "feature_mode" in obj:
            self.feature_mode = obj["feature_mode"]
        if "logit_eps" in obj:
            self.logit_eps = float(obj["logit_eps"])


# Convenience function to load a full Combiner with all base and meta weights

def load_combiner(
    xgb0_name,
    xgb1_name,
    meta_path,
    rob0_dir=None,
    rob1_dir=None,
    rob_dir=None,
) -> Combiner:
    """
    Load both XGB models, both single-task RoBERTa models, and meta-weights.

    Preferred args:
    - rob0_dir: directory for target_label='label0' RoBERTa
    - rob1_dir: directory for target_label='label1' RoBERTa

    Backward compatibility:
    - rob_dir can be used as shorthand to set both directories to the same value.
    """
    c = Combiner()
    c.xgb0.load_model(xgb0_name)
    c.xgb1.load_model(xgb1_name)
    if rob_dir is not None:
        rob0_dir = rob0_dir or rob_dir
        rob1_dir = rob1_dir or rob_dir
    if rob0_dir is None or rob1_dir is None:
        raise ValueError("Both rob0_dir and rob1_dir are required for combiner_both.")
    c.rob0.load_model("unused", model_dir=rob0_dir)
    c.rob1.load_model("unused", model_dir=rob1_dir)
    c.load(meta_path)
    return c

# Example usage:

"""
from src.combine_output.combiner_both import Combiner, load_combiner

combiner = load_combiner(
    xgb0_name="xgb0_combiner_model",
    xgb1_name="xgb1_combiner_model",
    meta_path="combiner.joblib",
    rob0_dir="./src/roberta/roberta_ai_model",
    rob1_dir="./src/roberta/roberta_bs_model",
)
"""
