"""
NLP measurement pipeline for HTRC accounting corpus.
Computes 6 text-derived measures designed to run inside the HTRC Data Capsule.
Only aggregate statistics are exported - fully HTRC-compliant.
"""
import csv
import math
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional


POSITIVE_WORDS = {
    "improve", "efficient", "accurate", "reliable", "consistent", "standardized",
    "transparent", "objective", "rigorous", "comprehensive", "effective", "sound"
}
NEGATIVE_WORDS = {
    "mislead", "inaccurate", "inconsistent", "problematic", "fraudulent",
    "inadequate", "misleading", "ambiguous", "deficient", "unreliable"
}
DISSENT_MARKERS = [
    "however", "contrary", "challenge", "dispute", "disagree",
    "criticism", "critique", "reject", "refute", "oppose",
    "inconsistent with", "problematic", "argue against", "at odds"
]
TOPIC_KEYWORDS = {
    "Auditing": ["audit", "auditor", "opinion", "attestation", "verification", "evidence", "sampling"],
    "Financial Reporting": ["financial statement", "balance sheet", "income statement", "disclosure", "footnote"],
    "Managerial Accounting": ["cost", "overhead", "variance", "budget", "standard cost", "absorption"],
    "Taxation": ["tax", "deduction", "depreciation", "amortization", "fiscal", "levy", "excise"],
    "Standards": ["GAAP", "IFRS", "standard", "principle", "rule", "regulation", "promulgate", "FASB"],
    "Theory": ["theory", "framework", "conceptual", "normative", "positive", "paradigm"]
}


class NLPPipeline:
    """Compute topic, sentiment, novelty, dissent, and text statistics for accounting articles."""

    def __init__(self):
        self._corpus_centroids: Dict[int, Counter] = {}

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    def classify_topic(self, tokens: List[str]) -> Dict[str, float]:
        token_set = set(tokens)
        scores = {
            topic: sum(1 for kw in kws if any(kw.lower() in t for t in token_set)) / max(len(kws), 1)
            for topic, kws in TOPIC_KEYWORDS.items()
        }
        total = sum(scores.values()) or 1
        return {t: round(s / total, 4) for t, s in scores.items()}

    def compute_sentiment(self, tokens: List[str]) -> Dict[str, float]:
        n = len(tokens) or 1
        pos = sum(1 for t in tokens if t in POSITIVE_WORDS) / n
        neg = sum(1 for t in tokens if t in NEGATIVE_WORDS) / n
        return {"sentiment_pos": round(pos, 5), "sentiment_neg": round(neg, 5),
                "sentiment_net": round(pos - neg, 5)}

    def compute_novelty(self, tokens: List[str], year: int) -> float:
        centroid = self._corpus_centroids.get(year)
        if not centroid:
            return 0.5
        doc_freq = Counter(tokens)
        dot = sum(doc_freq.get(t, 0) * centroid[t] for t in centroid)
        doc_norm = math.sqrt(sum(v ** 2 for v in doc_freq.values())) or 1
        cent_norm = math.sqrt(sum(v ** 2 for v in centroid.values())) or 1
        return round(1 - dot / (doc_norm * cent_norm), 4)

    def update_centroid(self, tokens: List[str], year: int):
        if year not in self._corpus_centroids:
            self._corpus_centroids[year] = Counter()
        self._corpus_centroids[year].update(tokens)

    def compute_dissent(self, tokens: List[str]) -> float:
        text = " ".join(tokens)
        hits = sum(text.count(m) for m in DISSENT_MARKERS)
        return round(hits / max(len(tokens), 1), 6)

    def compute_text_stats(self, tokens: List[str]) -> Dict[str, Any]:
        n = len(tokens) or 1
        avg_len = sum(len(t) for t in tokens) / n
        return {
            "word_count": n,
            "type_token_ratio": round(len(set(tokens)) / n, 4),
            "readability_proxy": round(max(0, 10 - avg_len), 3)
        }

    def process_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        text = article.get("text") or " ".join(article.get("tokens") or [])
        tokens = self.tokenize(text) if text else []
        if not tokens:
            return {**article, "error": "no_text"}

        year = int(article.get("year") or 1950)
        topics = self.classify_topic(tokens)
        primary = max(topics, key=topics.get)
        result = {
            **article,
            "primary_topic": primary,
            "topic_confidence": topics[primary],
            **self.compute_sentiment(tokens),
            "novelty_score": self.compute_novelty(tokens, year - 1),
            "dissent_rate": self.compute_dissent(tokens),
            **self.compute_text_stats(tokens)
        }
        self.update_centroid(tokens, year)
        return result

    def compute_measures(self, segments_csv: Path, ef_data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        results = []
        if not segments_csv.exists():
            print(f"Segments file not found: {segments_csv}")
            return results
        with open(segments_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                article = dict(row)
                if ef_data_dir:
                    import json
                    ef_file = ef_data_dir / f"{article.get('htid', '').replace('/', '_')}.json"
                    if ef_file.exists():
                        with open(ef_file) as jf:
                            ef = json.load(jf)
                        ps = int(article.get("page_start") or 0)
                        pe = int(article.get("page_end") or -1)
                        pages = ef.get("pages", [])
                        pe = len(pages) - 1 if pe == -1 else pe
                        article["tokens"] = list(
                            {k for p in pages[ps:pe + 1] for k in p.get("tokens", {}).keys()}
                        )
                results.append(self.process_article(article))
        print(f"  NLP measures computed for {len(results)} articles")
        return results

    def save_results(self, results: List[Dict[str, Any]], output_path: Path):
        if not results:
            return
        output_path.parent.mkdir(exist_ok=True)
        priority = [
            "htid", "article_id", "year", "journal", "page_start", "page_end",
            "primary_topic", "topic_confidence", "sentiment_pos", "sentiment_neg",
            "sentiment_net", "novelty_score", "dissent_rate",
            "word_count", "type_token_ratio", "segmentation_method"
        ]
        all_keys = {k for r in results for k in r}
        fields = priority + [k for k in sorted(all_keys) if k not in priority and k not in ("tokens", "text")]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
        print(f"  NLP results saved to {output_path}")
