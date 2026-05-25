"""
Multi-layer classifier for identifying accounting-related texts in HathiTrust corpus.
Combines metadata matching, Dewey classification, keyword presence, and EF token signals.
"""
import re
import csv
from typing import List, Dict, Any
from pathlib import Path


ACCOUNTING_KEYWORDS = {
    "core": [
        "accounting", "accountancy", "auditor", "auditing", "bookkeeping",
        "ledger", "journal entry", "double-entry", "balance sheet",
        "depreciation", "amortization", "accrual", "debit", "credit",
        "profit and loss", "income statement", "financial statement"
    ],
    "standards": [
        "GAAP", "IFRS", "FASB", "AICPA", "IASC", "accounting principles",
        "generally accepted", "accounting standards", "standard-setting"
    ],
    "managerial": [
        "cost accounting", "cost allocation", "overhead", "variance analysis",
        "budgeting", "standard cost", "absorption costing", "marginal cost"
    ],
    "audit": [
        "audit opinion", "qualified opinion", "going concern", "materiality",
        "internal control", "substantive testing", "audit evidence"
    ]
}

ACCOUNTING_DEWEY = ["657", "658.15", "332", "336"]

ACCOUNTING_JOURNALS = [
    "accounting review", "journal of accountancy", "the accountant",
    "cost and management", "management accounting", "accounting horizons",
    "journal of accounting", "auditing", "contemporary accounting"
]


class AccountingClassifier:
    """Classify HathiTrust volumes as accounting-related or not."""

    def __init__(self, confidence_threshold: float = 0.6):
        self.threshold = confidence_threshold
        self._all_kw = [kw for kws in ACCOUNTING_KEYWORDS.values() for kw in kws]

    def classify_volume(self, volume: Dict[str, Any]) -> Dict[str, Any]:
        score = 0.0
        signals = []

        title = (volume.get("title") or "").lower()
        journal = (volume.get("journal") or volume.get("source") or "").lower()
        for j in ACCOUNTING_JOURNALS:
            if j in title or j in journal:
                score += 0.4
                signals.append(f"journal:{j}")
                break

        subject = (volume.get("subject") or "").lower()
        sample = (volume.get("content_sample") or "").lower()
        text = f"{title} {subject} {sample}"
        hits = sum(1 for kw in self._all_kw if kw.lower() in text)
        if hits:
            score += min(hits * 0.05, 0.4)
            signals.append(f"kw_hits:{hits}")

        dewey = str(volume.get("dewey") or "")
        for prefix in ACCOUNTING_DEWEY:
            if dewey.startswith(prefix):
                score += 0.3
                signals.append(f"dewey:{dewey}")
                break

        if "ef_top_tokens" in volume:
            token_hits = sum(1 for kw in self._all_kw
                             if any(kw in t for t in volume["ef_top_tokens"]))
            if token_hits >= 3:
                score += min(token_hits * 0.04, 0.2)
                signals.append(f"ef_tokens:{token_hits}")

        confidence = min(score, 1.0)
        return {
            **volume,
            "is_accounting": confidence >= self.threshold,
            "confidence": round(confidence, 3),
            "classification_signals": signals
        }

    def classify_batch(self, volumes: List[Dict[str, Any]], verbose: bool = True) -> List[Dict[str, Any]]:
        results = []
        for i, vol in enumerate(volumes):
            results.append(self.classify_volume(vol))
            if verbose and (i + 1) % 100 == 0:
                n = sum(1 for r in results if r["is_accounting"])
                print(f"  Classified {i+1}/{len(volumes)} — {n} accounting texts")
        return results

    def save_results(self, results: List[Dict[str, Any]], output_path: Path):
        output_path.parent.mkdir(exist_ok=True)
        fieldnames = ["htid", "title", "journal", "year", "is_accounting",
                      "confidence", "classification_signals"]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in results:
                r["classification_signals"] = "|".join(r.get("classification_signals") or [])
                w.writerow(r)
        print(f"  Saved {len(results)} classified volumes to {output_path}")
