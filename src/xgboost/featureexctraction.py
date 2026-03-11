import pandas as pd
import numpy as np
import numpy.typing as npt


class LexiconFeatureExtractor:
    def __init__(self, lexicon_path: str):
        self.lexicon_df = pd.read_excel(lexicon_path)
        self.abstract_score = {
            word: (conc_m, conc_sd)
            for word, conc_m, conc_sd in zip(
                self.lexicon_df["Word"],
                self.lexicon_df["Conc.M"],
                self.lexicon_df["Conc.SD"],
            )
        }

    def extract_concreteness(self, feature_list: npt.NDArray[np.str_]):
        """
        Get mean concreteness and standard deviation for each feature token.
        Missing tokens use global mean fallbacks.

        :param self: Self
        :param feature_list: Array of features used by tfidf_vectorizer
        :type feature_list: npt.NDArray[np.str_]
        :return: numpy vector of means and standard deviations
        :rtype: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        """
        
        mean_filler = self.lexicon_df['Conc.M'].mean()
        sd_filler = self.lexicon_df['Conc.SD'].mean()

        means = np.zeros(len(feature_list))
        sds = np.zeros(len(feature_list))  

        for i, word in enumerate(feature_list):
            if word in self.abstract_score:
                means[i] = self.abstract_score[word][0]
                sds[i] = self.abstract_score[word][1]
            else:
                means[i] = mean_filler
                sds[i] = sd_filler
        means = (means - 1)/4
        sds = sds / 2
        return means, sds

    # Backward-compatible alias for existing callers.
    def extract_concretness(self, feature_list: npt.NDArray[np.str_]):
        return self.extract_concreteness(feature_list)


# Backward-compatible class alias for existing imports.
LexcionFeatureExtractor = LexiconFeatureExtractor
