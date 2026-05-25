"""
Article-level segmentation for HathiTrust journal volumes.
Primary: TOC + page-break detection from EF features.
Fallback: Accountants Index cross-reference, then page-level.
"""
import re
import csv
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


ARTICLE_START_PATTERNS = [
    r"^[A-Z][A-Z\s]{10,}$",     # ALL-CAPS title
    r"^By\s+[A-Z][a-z]+\s+[A-Z]",  # "By Author Name"
    r"^Abstract[:\.\s]",
    r"^Introduction\b",
]

DISSENT_MARKERS = [
    "however", "contrary", "challenge", "dispute", "disagree",
    "criticism", "critique", "reject", "refute", "oppose",
    "inconsistent", "problematic", "argue against"
]


class ArticleSegmenter:

    def __init__(self, min_article_pages: int = 3, max_article_pages: int = 60):
        self.min_pages = min_article_pages
        self.max_pages = max_article_pages

    def detect_article_boundaries(self, pages: List[Dict]) -> List[Tuple[int, int]]:
        """Detect article start/end indices from EF page data."""
        boundaries = [0]
        for i, page in enumerate(pages[1:], 1):
            tokens = page.get("tokens", {})
            page_text = " ".join(tokens.keys()) if isinstance(tokens, dict) else ""
            is_boundary = any(
                re.search(p, page_text, re.MULTILINE | re.IGNORECASE)
                for p in ARTICLE_START_PATTERNS
            )
            if not is_boundary and page.get("token_count", 999) < 50:
                is_boundary = True
            if is_boundary and i - boundaries[-1] >= self.min_pages:
                boundaries.append(i)

        segments = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] - 1 if i + 1 < len(boundaries) else len(pages) - 1
            if end - start >= self.min_pages - 1:
                segments.append((start, end))
        return segments

    def match_accountants_index(
        self, htid: str, year: int, volume: int, index_data: List[Dict]
    ) -> List[Dict]:
        return [
            {"htid": htid, "article_title": e.get("title"), "author": e.get("author"),
             "page_start": e.get("page_start"), "page_end": e.get("page_end"),
             "source": "accountants_index"}
            for e in index_data
            if e.get("year") == year and str(e.get("volume", "")) == str(volume)
        ]

    def segment_volumes(
        self,
        classified_csv: Path,
        ef_data_dir: Optional[Path] = None,
        index_data: Optional[List[Dict]] = None
    ) -> List[Dict]:
        results = []
        if not classified_csv.exists():
            print(f"Input not found: {classified_csv}")
            return results

        with open(classified_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("is_accounting", "")).lower() != "true":
                    continue
                htid = row["htid"]
                year = int(row.get("year") or 0)
                vol_num = int(row.get("volume") or 0)

                if ef_data_dir:
                    import json
                    ef_file = ef_data_dir / f"{htid.replace('/', '_')}.json"
                    if ef_file.exists():
                        with open(ef_file) as jf:
                            ef = json.load(jf)
                        segs = self.detect_article_boundaries(ef.get("pages", []))
                        for idx, (s, e) in enumerate(segs):
                            results.append({
                                "htid": htid, "article_id": f"{htid}_{idx:04d}",
                                "page_start": s, "page_end": e, "page_count": e - s + 1,
                                "year": year, "journal": row.get("journal", ""),
                                "segmentation_method": "ef_toc"
                            })
                        continue

                if index_data:
                    matches = self.match_accountants_index(htid, year, vol_num, index_data)
                    if matches:
                        for m in matches:
                            results.append({**m, "year": year, "segmentation_method": "accountants_index"})
                        continue

                results.append({
                    "htid": htid, "article_id": f"{htid}_full",
                    "page_start": 0, "page_end": -1, "page_count": -1,
                    "year": year, "journal": row.get("journal", ""),
                    "segmentation_method": "volume_level"
                })

        print(f"  Segmented {len(results)} articles/segments")
        return results

    def save_segments(self, segments: List[Dict], output_path: Path):
        if not segments:
            return
        output_path.parent.mkdir(exist_ok=True)
        fields = ["htid", "article_id", "page_start", "page_end", "page_count",
                  "year", "journal", "segmentation_method", "article_title", "author"]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(segments)
