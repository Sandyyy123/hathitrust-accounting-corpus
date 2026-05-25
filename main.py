#!/usr/bin/env python3
"""
HathiTrust Accounting Corpus Extractor
Entry point: identify, segment, and analyze historical accounting texts via HTRC Data Capsule.
"""
import argparse
import sys
from pathlib import Path
from accounting_classifier import AccountingClassifier
from article_segmenter import ArticleSegmenter
from nlp_pipeline import NLPPipeline
from htrc_connector import HTRCConnector


def main():
    parser = argparse.ArgumentParser(description="HTRC Accounting Corpus Analysis Pipeline")
    parser.add_argument("--mode", choices=["classify", "segment", "analyze", "full"], default="full")
    parser.add_argument("--input", help="HTRC Workset ID or volume list path")
    parser.add_argument("--output", default="output/", help="Output directory")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic demo data")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    if args.demo:
        print("Running in demo mode with synthetic data...")
        run_demo(output_dir)
        return

    connector = HTRCConnector()

    if args.mode in ("classify", "full"):
        print("Phase 1: Identifying accounting texts...")
        classifier = AccountingClassifier()
        volumes = connector.get_workset(args.input) if args.input else connector.search_accounting_texts()
        classified = classifier.classify_batch(volumes)
        classifier.save_results(classified, output_dir / "classified_volumes.csv")
        n_acct = sum(1 for v in classified if v["is_accounting"])
        print(f"  -> {n_acct} accounting volumes identified from {len(classified)} total")

    if args.mode in ("segment", "full"):
        print("Phase 2: Article-level segmentation...")
        segmenter = ArticleSegmenter()
        segments = segmenter.segment_volumes(output_dir / "classified_volumes.csv")
        segmenter.save_segments(segments, output_dir / "article_segments.csv")
        print(f"  -> {len(segments)} articles/segments identified")

    if args.mode in ("analyze", "full"):
        print("Phase 3: NLP analysis...")
        pipeline = NLPPipeline()
        results = pipeline.compute_measures(output_dir / "article_segments.csv")
        pipeline.save_results(results, output_dir / "nlp_measures.csv")
        print(f"  -> NLP measures computed for {len(results)} segments")

    print(f"\nDone. Results in {output_dir}/")


def run_demo(output_dir: Path):
    """Synthetic demo data showing output format."""
    import pandas as pd
    import random

    random.seed(42)
    accounting_keywords = ["audit", "ledger", "depreciation", "revenue", "GAAP", "balance sheet"]
    journals = ["The Accounting Review", "Journal of Accountancy", "The Accountant", "Cost and Management"]

    vols = []
    for i in range(60):
        is_acct = random.random() > 0.3
        vols.append({
            "htid": f"mdp.{39015000000000 + i}",
            "journal": random.choice(journals) if is_acct else f"General Journal {i}",
            "year": random.randint(1920, 1975),
            "volume": random.randint(1, 50),
            "is_accounting": is_acct,
            "confidence": round(random.uniform(0.7, 0.99), 3) if is_acct else round(random.uniform(0.01, 0.4), 3),
            "classification_signals": "|".join(random.sample(accounting_keywords, k=2)) if is_acct else ""
        })
    pd.DataFrame(vols).to_csv(output_dir / "classified_volumes_demo.csv", index=False)

    topics = ["Auditing", "Financial Reporting", "Managerial Accounting", "Taxation", "Standards"]
    nlp = []
    for i, v in enumerate([x for x in vols if x["is_accounting"]][:25]):
        nlp.append({
            "htid": v["htid"], "article_id": f"art_{i:04d}",
            "journal": v["journal"], "year": v["year"],
            "page_start": random.randint(1, 40), "page_end": random.randint(41, 80),
            "primary_topic": random.choice(topics),
            "topic_confidence": round(random.uniform(0.5, 0.95), 3),
            "sentiment_pos": round(random.uniform(0.05, 0.45), 3),
            "sentiment_neg": round(random.uniform(0.02, 0.25), 3),
            "novelty_score": round(random.uniform(0.1, 0.9), 3),
            "dissent_rate": round(random.uniform(0.001, 0.05), 4),
            "word_count": random.randint(2500, 12000),
            "type_token_ratio": round(random.uniform(0.38, 0.68), 3),
        })
    pd.DataFrame(nlp).to_csv(output_dir / "nlp_measures_demo.csv", index=False)

    print(f"Demo output written to {output_dir}/")
    print(f"  classified_volumes_demo.csv: {len(vols)} volumes ({sum(1 for v in vols if v['is_accounting'])} accounting)")
    print(f"  nlp_measures_demo.csv: {len(nlp)} articles with 6 NLP measures each")


if __name__ == "__main__":
    main()
