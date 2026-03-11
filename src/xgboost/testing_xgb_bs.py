from pathlib import Path

from src.xgboost.xgb_model import XGBModel
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGED_DIR = REPO_ROOT / "data_cleansing" / "merged"

df_train = pd.read_csv(MERGED_DIR / "bullshit_training_all.csv")

texts = df_train["text"].to_numpy()
label1 = df_train["label1"].to_numpy()

xgb = XGBModel()

df_test = pd.read_csv(MERGED_DIR / "bullshit_testing_all.csv")

texts_test = df_test["text"].to_numpy()
label1_test = df_test["label1"].to_numpy()


def evaluate(model, texts, labels, label_names=None):
    x_tfidf = model.vectorizer.transform(texts)

    # Handle vagueness model's extra features if they exist
    if hasattr(model, 'means'):
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
    tick_labels = label_names if label_names is not None else sorted(set(labels))  # fix here
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=tick_labels, yticklabels=tick_labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(str(Path(__file__).resolve().parent / "confusion_matrix_xgb_bs.png"))
    plt.show()


print("=== CV Accuracy ===")
print(xgb.train_validate_new_model(texts, label1))

print("\n=== Test Set ===")
evaluate(xgb, texts_test, label1_test)


# Without vagueness features:

""" - OLD DATA SET
train acc: 0.9728771927050184
test acc: 0.9188281709057332
"""

""" - NEW, MORE BALANCED DATA SET
train acc: 0.9676316921021122
test acc: 0.9703225806451613

no nums:
0.9647038044017544
0.9690322580645161
"""

# With vagueness features:

""" - untuned model
train acc: 0.7310528091448532
test acc: 0.8129032258064516
"""

""" - tuned model
just kfold and classifier params:
0.9622629277553963
0.9638709677419355

with vectorizer params (including nums):
0.9682820986061772
0.9716129032258064

no nums:
0.9647034074897298
0.9703225806451613
"""
