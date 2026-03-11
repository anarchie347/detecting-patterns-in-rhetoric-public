from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.xgboost.xgb_model import XGBModel

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGED_DIR = REPO_ROOT / "data_cleansing" / "merged"

df_train = pd.read_csv(MERGED_DIR / "ai_training_all.csv")
texts = df_train["text"].astype(str).to_numpy()
label0 = df_train["label0"].astype(np.uint8).to_numpy()

xgb = XGBModel()

df_test = pd.read_csv(MERGED_DIR / "ai_testing_all.csv")
texts_test = df_test["text"].astype(str).to_numpy()
label0_test = df_test["label0"].astype(np.uint8).to_numpy()


def evaluate(model, texts, labels, label_names=None):
    x_tfidf = model.vectorizer.transform(texts)

    if hasattr(model, "means"):
        from scipy.sparse import hstack

        total_weight = np.array(x_tfidf.sum(axis=1)) + 1e-9
        x_mean = (x_tfidf.dot(model.means).reshape(-1, 1)) / total_weight
        x_sd = (x_tfidf.dot(model.sds).reshape(-1, 1)) / total_weight
        X = hstack([x_tfidf, x_mean, x_sd])
    else:
        X = x_tfidf

    preds = model.classifer.predict(X)
    print(classification_report(labels, preds, target_names=label_names))

    cm = confusion_matrix(labels, preds)
    tick_labels = label_names if label_names is not None else sorted(set(labels))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=tick_labels, yticklabels=tick_labels)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(str(Path(__file__).resolve().parent / "confusion_matrix_xgb_ai.png"))
    plt.show()


print("=== CV Accuracy ===")
print(xgb.train_validate_new_model(texts, label0))

print("\n=== Test Set ===")
evaluate(xgb, texts_test, label0_test)


"""
=== CV Accuracy ===
0.9582245528094443

=== Test Set ===
              precision    recall  f1-score   support

           0       0.84      0.98      0.90       279
           1       0.98      0.81      0.88       279

    accuracy                           0.89       558
   macro avg       0.91      0.89      0.89       558
weighted avg       0.91      0.89      0.89       558
"""