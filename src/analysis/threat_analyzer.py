"""
Real-time threat analyzer, confidence scoring, and Explainable AI (XAI) engine.
Combines statistical NLP model predictions with cybersecurity heuristic rules.
"""

from typing import Dict, Any, List, Tuple
import numpy as np

from src.preprocessing.text_cleaner import EmailTextCleaner
from src.features.vectorizer import EmailTFIDFVectorizer
from src.models.classifier import PhishingClassifier


class ThreatAnalyzer:
    """
    Comprehensive cybersecurity threat analysis engine.
    Analyzes email payloads, evaluates posterior probabilities, identifies malicious vectors,
    and returns token-level explainability for human analyst review.
    """

    def __init__(
        self,
        cleaner: EmailTextCleaner,
        vectorizer: EmailTFIDFVectorizer,
        classifier: PhishingClassifier,
    ):
        self.cleaner = cleaner
        self.vectorizer = vectorizer
        self.classifier = classifier

    def analyze(self, raw_text: str, subject: str = "") -> Dict[str, Any]:
        """
        Perform complete end-to-end threat analysis on incoming email.
        """
        full_text = f"{subject}\n{raw_text}".strip() if subject else raw_text.strip()
        if not full_text:
            return self._empty_analysis()

        # 1. Preprocessing & Heuristic Extraction
        cleaned_text, heuristics = self.cleaner.process(full_text)

        # 2. Vectorization
        features = self.vectorizer.transform([cleaned_text])

        # 3. Model Inference
        probabilities = self.classifier.predict_proba(features)[0]
        prob_ham = float(probabilities[0])
        prob_phish = float(probabilities[1])
        prediction = int(np.argmax(probabilities))  # 0: Legitimate, 1: Phishing

        # 4. Composite Threat Score (0.0 to 100.0)
        # Baseline is model probability
        threat_score = prob_phish * 100.0

        # Heuristic boost for egregious threats (IP URLs, high caps + urgent keywords)
        if heuristics["has_ip_url"]:
            threat_score = max(threat_score, 88.0)
        if heuristics["has_urgent_keywords"] and heuristics["caps_ratio"] > 0.25 and threat_score > 40:
            threat_score = min(100.0, threat_score + 10.0)

        threat_score = round(min(100.0, max(0.0, threat_score)), 1)

        # 5. Threat Severity Tier
        if threat_score < 25.0:
            severity = "SAFE"
            verdict = "Legitimate Email"
            color_theme = "emerald"
            badge = "safe"
        elif threat_score < 50.0:
            severity = "LOW RISK"
            verdict = "Likely Legitimate (Low Risk)"
            color_theme = "blue"
            badge = "low"
        elif threat_score < 75.0:
            severity = "SUSPICIOUS"
            verdict = "Suspicious Email (Caution Advised)"
            color_theme = "amber"
            badge = "suspicious"
        else:
            severity = "CRITICAL THREAT"
            verdict = "Phishing Email Detected"
            color_theme = "rose"
            badge = "phishing"

        # 6. Explainable AI: Token Attribution
        top_tokens = self.vectorizer.get_top_features_for_text(cleaned_text, top_n=15)
        feature_names = self.vectorizer.get_feature_names()
        word_signals = self._compute_token_attributions(cleaned_text)

        # 7. Actionable Security Recommendations
        recommendations = self._generate_recommendations(heuristics, threat_score, severity)

        return {
            "prediction": "Phishing" if prediction == 1 or threat_score >= 50.0 else "Legitimate",
            "verdict": verdict,
            "severity": severity,
            "threat_score": threat_score,
            "badge": badge,
            "color_theme": color_theme,
            "probabilities": {
                "phishing": round(prob_phish * 100.0, 2),
                "legitimate": round(prob_ham * 100.0, 2),
            },
            "heuristics": heuristics,
            "explainability": {
                "top_tfidf_features": top_tokens,
                "token_signals": word_signals,
            },
            "recommendations": recommendations,
        }

    def _compute_token_attributions(self, cleaned_text: str) -> List[Dict[str, Any]]:
        """
        Determine whether specific words in the email push toward Phishing or Legitimate.
        """
        if not cleaned_text or self.classifier._raw_nb is None:
            return []

        tokens = list(set(cleaned_text.split()))
        feature_names = self.vectorizer.get_feature_names()
        vocab_map = {name: idx for idx, name in enumerate(feature_names)}

        ham_log_probs = self.classifier._raw_nb.feature_log_prob_[0]
        phish_log_probs = self.classifier._raw_nb.feature_log_prob_[1]

        signals = []
        for t in tokens:
            if t in vocab_map:
                idx = vocab_map[t]
                log_ratio = float(phish_log_probs[idx] - ham_log_probs[idx])
                category = "phishing" if log_ratio > 0.3 else ("legitimate" if log_ratio < -0.3 else "neutral")
                signals.append({
                    "word": t,
                    "impact": round(abs(log_ratio), 2),
                    "tendency": category,
                    "score": round(log_ratio, 2)
                })

        # Sort by absolute impact
        signals.sort(key=lambda x: x["impact"], reverse=True)
        return signals[:12]

    def _generate_recommendations(self, heuristics: Dict[str, Any], threat_score: float, severity: str) -> List[str]:
        recs = []
        if heuristics.get("has_ip_url"):
            recs.append("CRITICAL: Detected links pointing to raw IP addresses. Never enter credentials on IP destinations.")
        if heuristics.get("urgent_keywords_found"):
            kws = ", ".join(heuristics["urgent_keywords_found"][:3])
            recs.append(f"Psychological Urgency Trigger: Contains pressure keywords ({kws}). Standard organizational procedure is to verify requests out-of-band.")
        if heuristics.get("money_references", 0) > 0:
            recs.append("Financial References: Verify any invoices, wire instructions, or gift card requests directly via official corporate phone channels.")
        if heuristics.get("caps_ratio", 0) > 0.2:
            recs.append("Excessive Capitalization: Often utilized in spear-phishing campaigns to induce alarm and bypass cognitive hesitation.")

        if severity in ["CRITICAL THREAT", "SUSPICIOUS"]:
            recs.append("Quarantine Action: Mark as phishing in your mail client and forward headers to the SOC / Security Operations team.")
        else:
            recs.append("Standard Protocol: Email exhibits authentic linguistic patterns and lacks high-confidence malicious signatures.")

        return recs

    def _empty_analysis(self) -> Dict[str, Any]:
        return {
            "prediction": "Unknown",
            "verdict": "No Content Provided",
            "severity": "SAFE",
            "threat_score": 0.0,
            "badge": "safe",
            "color_theme": "emerald",
            "probabilities": {"phishing": 0.0, "legitimate": 100.0},
            "heuristics": self.cleaner.extract_heuristics(""),
            "explainability": {"top_tfidf_features": [], "token_signals": []},
            "recommendations": ["Enter or paste email body to execute threat analysis."],
        }
