"""
Comprehensive unit and integration test suite for Phishing Email Classifier & Web Application.
Uses Python standard library unittest for zero-dependency execution.
"""

import os
import sys
import unittest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.text_cleaner import EmailTextCleaner
from src.features.vectorizer import EmailTFIDFVectorizer
from src.models.classifier import PhishingClassifier
from src.analysis.threat_analyzer import ThreatAnalyzer
from app import app


class TestPhishingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cleaner = EmailTextCleaner(remove_stopwords=True)
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        model_path = os.path.join(models_dir, "naive_bayes_model.joblib")
        vec_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")

        if not os.path.exists(model_path) or not os.path.exists(vec_path):
            from src.models.train import train_and_evaluate
            train_and_evaluate()

        cls.vectorizer = EmailTFIDFVectorizer.load(vec_path)
        cls.classifier = PhishingClassifier.load(model_path)
        cls.analyzer = ThreatAnalyzer(cls.cleaner, cls.vectorizer, cls.classifier)

        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_01_html_stripping(self):
        raw = "<html><body><h1>Security Alert</h1><p>Click here immediately</p></body></html>"
        cleaned = self.cleaner.clean_text(raw)
        self.assertNotIn("<", cleaned)
        self.assertNotIn(">", cleaned)
        self.assertIn("security", cleaned)
        self.assertIn("alert", cleaned)

    def test_02_url_and_ip_tokenization(self):
        raw = "Verify your account at http://192.168.1.1/login or visit https://paypal.com"
        cleaned = self.cleaner.clean_text(raw)
        self.assertIn("token_ip_url", cleaned)
        self.assertIn("token_url", cleaned)

    def test_03_heuristics_extraction(self):
        raw = "URGENT ACTION REQUIRED! Wire $5,000 to http://10.0.0.1/auth"
        h = self.cleaner.extract_heuristics(raw)
        self.assertTrue(h["has_ip_url"])
        self.assertGreaterEqual(h["money_references"], 1)
        self.assertGreater(h["caps_ratio"], 0.3)
        self.assertTrue(h["has_urgent_keywords"])

    def test_04_phishing_detection(self):
        phish_text = """Subject: URGENT: Your PayPal Account Has Been Suspended
Dear Customer,
We detected unauthorized login attempts from Russia.
Click here immediately to restore access: http://192.168.1.10/login.php
Failure to verify within 24 hours will result in permanent account termination."""
        
        result = self.analyzer.analyze(phish_text)
        self.assertEqual(result["prediction"], "Phishing")
        self.assertGreater(result["threat_score"], 80.0)
        self.assertIn(result["severity"], ["CRITICAL THREAT", "SUSPICIOUS"])
        self.assertGreater(len(result["recommendations"]), 0)

    def test_05_legitimate_detection(self):
        ham_text = """Subject: Sprint 24 Retrospective & Q4 Roadmap Planning
Hi team,
Attached is the revised agenda for our bi-weekly sprint planning session tomorrow at 10 AM.
Key discussion points:
- Milestone timeline and release dates
- Performance testing results
- Resource allocation across squads
Best regards,
Alex Mercer"""

        result = self.analyzer.analyze(ham_text)
        self.assertEqual(result["prediction"], "Legitimate")
        self.assertLess(result["threat_score"], 40.0)
        self.assertIn(result["severity"], ["SAFE", "LOW RISK"])

    def test_06_flask_index(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Phishing Email Classifier", res.data)

    def test_07_flask_metrics(self):
        res = self.client.get("/metrics")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Model Performance", res.data)

    def test_08_flask_health(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "healthy")

    def test_09_api_scan_phishing(self):
        payload = {
            "subject": "URGENT: Microsoft 365 Password Expiration",
            "body": "Your password expires in 24 hours. Update immediately at http://microsoft-portal-login.top/auth"
        }
        res = self.client.post("/api/scan", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["analysis"]["prediction"], "Phishing")
        self.assertGreater(data["analysis"]["threat_score"], 50)

    def test_10_api_presets(self):
        res = self.client.get("/api/presets")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 4)


if __name__ == "__main__":
    unittest.main()
