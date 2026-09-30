"""
TF-IDF Feature extraction and vectorization pipeline.
Extracts unigram and bigram features with sublinear term-frequency scaling.
"""

import os
from typing import List, Tuple
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class EmailTFIDFVectorizer:
    """
    TF-IDF Vectorizer wrapper specialized for phishing detection.
    Captures unigrams and high-discriminative bigrams (e.g., 'account suspended', 'wire transfer').
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        min_df: int = 2,
        max_df: float = 0.95,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.max_df = max_df

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        self.is_fitted = False

    def fit(self, texts: List[str]):
        """Fit the TF-IDF vectorizer on a corpus of cleaned email strings."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]):
        """Transform texts into sparse TF-IDF feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("Vectorizer must be fitted before calling transform.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]):
        """Fit vectorizer and return transformed TF-IDF matrix."""
        features = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return features

    def get_feature_names(self) -> List[str]:
        """Return the list of vocabulary feature names."""
        return self.vectorizer.get_feature_names_out().tolist()

    def get_top_features_for_text(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract the most heavily weighted TF-IDF tokens for an individual email text.
        Useful for explainable AI and user threat inspection.
        """
        if not self.is_fitted:
            return []

        tfidf_vec = self.vectorizer.transform([text]).toarray()[0]
        feature_names = self.vectorizer.get_feature_names_out()

        # Find non-zero indices
        non_zero_indices = np.where(tfidf_vec > 0)[0]
        if len(non_zero_indices) == 0:
            return []

        scored_features = [(feature_names[i], float(tfidf_vec[i])) for i in non_zero_indices]
        scored_features.sort(key=lambda x: x[1], reverse=True)
        return scored_features[:top_n]

    def save(self, filepath: str):
        """Persist fitted vectorizer artifact to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self.vectorizer, filepath)

    @classmethod
    def load(cls, filepath: str) -> "EmailTFIDFVectorizer":
        """Load persisted vectorizer artifact from disk."""
        instance = cls()
        instance.vectorizer = joblib.load(filepath)
        instance.is_fitted = True
        return instance
