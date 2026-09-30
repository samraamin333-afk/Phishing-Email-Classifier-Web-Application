"""
Text preprocessing and feature normalization pipeline for email text.
Handles HTML removal, URL/IP tokenization, header extraction, and linguistic cleaning.
"""

import re
import html
from typing import Dict, Any, Tuple, List


class EmailTextCleaner:
    """
    NLP Preprocessor specifically tailored for cybersecurity email classification.
    Cleans raw email payloads while extracting security-critical heuristic indicators.
    """

    # High-risk phishing trigger phrases
    URGENT_KEYWORDS = [
        "account suspended", "verify your account", "immediate action required",
        "action required", "unauthorized access", "security alert", "password reset",
        "update billing", "confirm identity", "wire transfer", "bank account",
        "suspended notice", "login immediately", "click here", "temporarily locked",
        "payroll update", "urgent invoice", "gift card", "crypto deposit",
        "tax refund", "irs notice", "limited time", "within 24 hours",
        "compromised", "validate credentials", "dear customer", "undelivered package"
    ]

    # Standard English stopwords (excluding security-sensitive tokens like 'not', 'no', 'update')
    STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
        "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
        "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
        "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
        "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
        "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
        "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
        "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
        "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
        "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
        "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
        "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
        "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when",
        "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
        "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
        "you're", "you've", "your", "yours", "yourself", "yourselves"
    }

    # Regex patterns
    URL_REGEX = re.compile(r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+|www\.[-\w.]+\.[a-zA-Z]{2,}", re.IGNORECASE)
    IP_URL_REGEX = re.compile(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?(?:/[^\s]*)?", re.IGNORECASE)
    EMAIL_REGEX = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+", re.IGNORECASE)
    HTML_REGEX = re.compile(r"<[^>]+>")
    MONEY_REGEX = re.compile(r"[\$\£\€]\s*[\d,]+(?:\.\d+)?|\b\d+\s*(?:dollars|usd|eur|gbp|btc|bitcoin)\b", re.IGNORECASE)
    HEX_OR_ENTITY = re.compile(r"&#?[a-zA-Z0-9]+;")

    def __init__(self, remove_stopwords: bool = True):
        self.remove_stopwords = remove_stopwords

    def extract_heuristics(self, raw_text: str) -> Dict[str, Any]:
        """
        Extract cyber-threat heuristic indicators from the raw email before text stripping.
        """
        if not raw_text:
            return {
                "url_count": 0,
                "has_ip_url": False,
                "has_urgent_keywords": False,
                "urgent_keywords_found": [],
                "caps_ratio": 0.0,
                "money_references": 0,
                "exclamation_count": 0,
                "raw_char_count": 0,
            }

        text_lower = raw_text.lower()

        # URLs and IPs
        urls = self.URL_REGEX.findall(raw_text)
        ip_urls = self.IP_URL_REGEX.findall(raw_text)

        # Keyword matching
        matched_keywords = [kw for kw in self.URGENT_KEYWORDS if kw in text_lower]

        # Financial mentions
        money_matches = self.MONEY_REGEX.findall(raw_text)

        # Casing analysis
        alpha_chars = [c for c in raw_text if c.isalpha()]
        upper_chars = [c for c in alpha_chars if c.isupper()]
        caps_ratio = len(upper_chars) / max(len(alpha_chars), 1)

        # Exclamations
        exclamation_count = raw_text.count("!")

        return {
            "url_count": len(urls),
            "has_ip_url": len(ip_urls) > 0,
            "has_urgent_keywords": len(matched_keywords) > 0,
            "urgent_keywords_found": matched_keywords,
            "caps_ratio": round(caps_ratio, 3),
            "money_references": len(money_matches),
            "exclamation_count": exclamation_count,
            "raw_char_count": len(raw_text),
        }

    def clean_text(self, text: str) -> str:
        """
        Execute full NLP cleaning pipeline on email text.
        Converts raw body to tokenized, normalized representation suitable for TF-IDF.
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Unescape HTML entities (&amp;, &#39;, etc.)
        cleaned = html.unescape(text)

        # 2. Strip HTML tags
        cleaned = self.HTML_REGEX.sub(" ", cleaned)

        # 3. Normalize IP URLs to high-risk token
        cleaned = self.IP_URL_REGEX.sub(" token_ip_url ", cleaned)

        # 4. Normalize standard URLs
        cleaned = self.URL_REGEX.sub(" token_url ", cleaned)

        # 5. Normalize Email addresses
        cleaned = self.EMAIL_REGEX.sub(" token_email ", cleaned)

        # 6. Normalize Monetary references
        cleaned = self.MONEY_REGEX.sub(" token_money ", cleaned)

        # 7. Convert to lowercase
        cleaned = cleaned.lower()

        # 8. Clean non-alphanumeric characters (keep tokens and spaces)
        cleaned = re.sub(r"[^a-z0-9_\s]", " ", cleaned)

        # 9. Tokenize & remove redundant stopwords
        tokens = cleaned.split()
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.STOPWORDS and len(t) > 1]

        # 10. Rejoin
        return " ".join(tokens)

    def process(self, raw_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process raw email text: returns (cleaned_text, heuristic_dict).
        """
        heuristics = self.extract_heuristics(raw_text)
        cleaned_text = self.clean_text(raw_text)
        heuristics["cleaned_char_count"] = len(cleaned_text)
        heuristics["compression_pct"] = round(
            (1.0 - (len(cleaned_text) / max(heuristics["raw_char_count"], 1))) * 100, 1
        )
        return cleaned_text, heuristics
