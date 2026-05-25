# hathitrust-accounting-corpus

Systematic pipeline for identifying, segmenting, and analyzing historical accounting texts in the HathiTrust corpus. Built for use within HTRC Data Capsules, fully compliant with HTRC export restrictions.

## Architecture

```
HathiTrust Corpus (16M+ volumes)
        |
        v
[AccountingClassifier]     <- metadata + keyword + Dewey + EF token signals
        | ~12K accounting volumes
        v
[ArticleSegmenter]         <- TOC detection -> Accountants' Index -> page-level fallback
        | article/page segments
        v
[NLPPipeline]              <- topic + sentiment + novelty + dissent + text stats
        | all computed inside HTRC Data Capsule
        v
Exported: aggregate CSV (no raw text exported -- HTRC-compliant)
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

```bash
# Demo mode (no HTRC credentials needed)
python main.py --demo --output output/

# Full pipeline with HTRC workset
python main.py --mode full --input YOUR_WORKSET_ID --output output/

# Individual phases
python main.py --mode classify --output output/
python main.py --mode segment --output output/
python main.py --mode analyze --output output/
```

## HTRC Data Capsule Notes

- All text-level NLP computation runs inside the capsule
- Only aggregate statistics and page-level derived measures are exported
- Public domain texts allow full text export; in-copyright texts use EF features only
- See `HTRC_CAPSULE_GUIDE.md` (delivered as project memo) for full setup instructions

## NLP Measures

| Measure | Method | Notes |
|---------|--------|-------|
| Topic classification | BERTopic / LDA keyword fallback | 6 accounting sub-domains |
| Sentiment | VADER + FinBERT inside capsule | Positive / negative / net |
| Novelty | Cosine distance from prior-year centroid | Higher = more novel vocabulary |
| Dissent | KWIC marker frequency | 13 dissent phrase markers |
| Text stats | Token count, TTR, readability | EF-derived, no raw text needed |
| Text length | Page count + word count | Direct from EF metadata |

## Author

Dr. Sandeep Grover | PhD Data Science | Computational Research
Charite Berlin - Lubeck - Tubingen - Bonn/Marburg
