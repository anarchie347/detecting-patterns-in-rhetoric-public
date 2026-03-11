from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
from pathlib import Path
from typing import cast
import numpy.typing as npt
from src.xgboost.featureexctraction import LexcionFeatureExtractor
from scipy.sparse import hstack


class XGBModel:
    def __init__(self) -> None:
        self.vectorizer : TfidfVectorizer | None = None
        self.classifer : XGBClassifier | None = None

    def _build_feature_matrix(self, x_tfidf):
        total_weight = np.array(x_tfidf.sum(axis=1)) + 1e-9
        x_mean = (x_tfidf.dot(self.means).reshape(-1, 1)) / total_weight
        x_sd = (x_tfidf.dot(self.sds).reshape(-1, 1)) / total_weight
        return hstack([x_tfidf, x_mean, x_sd])


    def train_validate_new_model(self, training_data : npt.NDArray[np.str_], labels : npt.NDArray[np.uint8], folds : int = 5) -> np.float64:
        """
        Performs k-fold cross validation on on a model, and returns mean accuracy. Default 5 folds. At end keeps the model from the last iteration of folding
        
        :param self: Self
        :param training_data: Array of training data strings
        :type training_data: npt.NDArray[np.str_]
        :param labels: Array of classes for training data
        :type labels: npt.NDArray[np.uint8]
        :param folds: number of folds, default 5
        :type folds: int
        :return: Mean accuracy across the k-fold validation
        :rtype: float64
        """
        # kfold = KFold(n_splits=5)
        kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
        accuracies = []

        for (train_index, val_index) in kfold.split(training_data):
            self.train_new_model(training_data[train_index], labels[train_index])
            a = self.test_model(training_data[val_index], labels[val_index])
            accuracies.append(a)
        return cast(np.float64, np.mean(accuracies))


    def train_new_model(self, training_data : npt.NDArray[np.str_], labels : npt.NDArray[np.uint8]) -> None:
        """
        Trains a new model based on all training data

        :param self: Self
        :param training_data: Array of training data strings
        :type training_data: npt.NDArray[np.str_]
        :param labels: Array of classes for training data
        :type labels: npt.NDArray[np.uint8]
        """
        # TF-IDF
        #https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        # may want to change this to a custom stop word list, 'maybe get rid of max_features?
        
        # tfidf_vectorizer = TfidfVectorizer(max_features=5000, lowercase=True, stop_words="english", 
        #                                    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b") # removes numbers
        tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),       # captures bigrams like "not good"
            sublinear_tf=True,        # log-scale TF, helps with frequent terms
            min_df=2,                 # ignore very rare terms
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b" # removes numbers
        )

        x_tfidf = tfidf_vectorizer.fit_transform(training_data) #perform TF-IDF
        self.vectorizer = tfidf_vectorizer

        feature_names = self.vectorizer.get_feature_names_out()
        lexicon_path = Path(__file__).resolve().parent / "Concreteness_ratings_Brysbaert_et_al_BRM.2.xlsx"
        lexicon_exctractor = LexcionFeatureExtractor(str(lexicon_path))
        self.means, self.sds = lexicon_exctractor.extract_concretness(feature_names)
        
        X = self._build_feature_matrix(x_tfidf)
        
        y = labels

        # self.classifer = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=1, verbosity =2)
        self.classifer = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, verbosity=2)
        self.classifer.fit(X, y)

    def test_model(self, testing_data : npt.NDArray[np.str_], labels : npt.NDArray[np.uint8]) -> np.float64:
        """
        Tests the currently loaded model using test data, returns accuracy
        
        :param self: Self
        :param testing_data: Array of testing data strings
        :type testing_data: npt.NDArray[np.str_]
        :param labels: Array of classes for training data
        :type labels: npt.NDArray[np.uint8]
        :return: Accuracy
        :rtype: float64
        """
        
        if (self.vectorizer is None or self.classifer is None):
            raise ValueError("No model to test")
        
        x_tfidf = self.vectorizer.transform(testing_data)
        
        X = self._build_feature_matrix(x_tfidf)
        preds = self.classifer.predict(X)

        return np.mean(preds == labels)  


    def load_model(self, model_save_name : str) -> None:
        """
        Loads model from a file given a name. 
        Files are <name>-xgb.json and <name>-vectorizer.pkl and <name>-lexicon.pkl
        
        :param self: Self
        :param model_save_name: Name for model files to load
        :type model_save_name: str
        """
        xgb_file = XGBModel.__name_to_filename_xgb(model_save_name)
        vec_file = XGBModel.__name_to_filename_vect(model_save_name)
        lexicon_file = f"{model_save_name}-lexicon.pkl"
        self.classifer = XGBClassifier()
        try:
            self.classifer.load_model(xgb_file)
        except:
            raise ValueError(f"File not found or wrong format: {xgb_file}")
        try:
            with open(vec_file, "rb") as f:
                self.vectorizer = cast(TfidfVectorizer, pickle.load(f))
        except:
            raise ValueError(f"File not found or wrong format: {vec_file}")
        try:
            with open(lexicon_file, "rb") as f:
                lexicon = pickle.load(f)
                self.means = lexicon["means"]
                self.sds = lexicon["sds"]
        except:
            raise ValueError(f"File not found or wrong format: {lexicon_file}")

    def save_model(self, model_save_name : str) -> bool:
        """
        Saves the current model to three files. 
        Files are <name>-xgb.json and <name>-vectorizer.pkl and <name>-lexicon.pkl
        
        :param self: Self
        :param model_save_name: Name for model files to load
        :type model_save_name: str
        :return: Whether or not there was a current model to save
        :rtype: bool
        """
        if (self.vectorizer is None or self.classifer is None):
            return False
        
        xgb_file = XGBModel.__name_to_filename_xgb(model_save_name)
        vec_file = XGBModel.__name_to_filename_vect(model_save_name)
        lexicon_file = f"{model_save_name}-lexicon.pkl"

        with open(vec_file, "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(lexicon_file, "wb") as f:
            pickle.dump({"means": self.means, "sds": self.sds}, f)
        self.classifer.save_model(xgb_file)
        return True
    

    @staticmethod
    def __name_to_filename_xgb(name : str) -> str:
        return f"{name}-xgb.json"
    @staticmethod
    def __name_to_filename_vect(name : str) -> str:
        return f"{name}-vectorizer.pkl"
