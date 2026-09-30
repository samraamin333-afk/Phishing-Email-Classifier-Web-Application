"""
Naive Bayes Classifier module for email threat classification.
Supports Multinomial Naive Bayes with feature log-odds explainability.
"""

import os
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV


class PhishingClassifier:
    """
    Multinomial Naive Bayes Classifier with probability calibration and
    feature log-odds ratio extraction for Explainable AI (XAI).
    """

    def __init__(self, alpha: float = 0.2, calibrate: bool = True):
        self.alpha = alpha
        self.calibrate = calibrate
        self.base_model = MultinomialNB(alpha=self.alpha)
        self.model = CalibratedClassifierCV(self.base_model, cv=5) if calibrate else self.base_model
        self.classes_ = [0, 1]  # 0: Legitimate (Ham), 1: Phishing
        self.class_names = {0: "Legitimate", 1: "Phishing"}
        self.is_fitted = False
        self._raw_nb: MultinomialNB = None

    def fit(self, X, y):
        """Train the Naive Bayes classifier on vectorized training features."""
        # Also fit a standalone raw MultinomialNB for feature log-prob explainability
        self._raw_nb = MultinomialNB(alpha=self.alpha)
        self._raw_nb.fit(X, y)

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict binary class labels (0 or 1)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Predict calibrated posterior class probabilities [[p_ham, p_phish], ...]."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict_proba(X)

    def get_top_phishing_features(self, feature_names: List[str], top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Compute top tokens with highest log-odds ratio favoring Phishing over Legitimate:
        log( P(word | Phishing) / P(word | Legitimate) ).
        """
        if self._raw_nb is None:
            return []

        # feature_log_prob_ has shape [n_classes, n_features]
        # class 0: Legitimate, class 1: Phishing
        ham_log_probs = self._raw_nb.feature_log_prob_[0]
        phish_log_probs = self._raw_nb.feature_log_prob_[1]

        log_odds = phish_log_probs - ham_log_probs
        top_indices = np.argsort(log_odds)[::-1][:top_n]

        results = [(feature_names[i], float(log_odds[i])) for i in top_indices]
        return results

    def get_top_ham_features(self, feature_names: List[str], top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Compute top tokens with highest log-odds ratio favoring Legitimate over Phishing:
        log( P(word | Legitimate) / P(word | Phishing) ).
        """
        if self._raw_nb is None:
            return []

        ham_log_probs = self._raw_nb.feature_log_prob_[0]
        phish_log_probs = self._raw_nb.feature_log_prob_[1]

        log_odds = ham_log_probs - phish_log_probs
        top_indices = np.argsort(log_odds)[::-1][:top_n]

        results = [(feature_names[i], float(log_odds[i])) for i in top_indices]
        return results

    def save(self, filepath: str):
        """Persist trained model artifacts."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "raw_nb": self._raw_nb,
            "alpha": self.alpha,
            "calibrate": self.calibrate,
            "is_fitted": self.is_fitted,
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "PhishingClassifier":
        """Load trained model artifacts from disk."""
        data = joblib.load(filepath)
        instance = cls(alpha=data.get("alpha", 0.2), calibrate=data.get("calibrate", True))
        instance.model = data["model"]
        instance._raw_nb = data.get("raw_nb", None)
        instance.is_fitted = data.get("is_fitted", True)
        return instance
