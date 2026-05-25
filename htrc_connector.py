"""
HTRC Data Capsule connector. Wraps htrc-feature-reader and HTRC Workset + catalog APIs.
"""
import requests
from typing import List, Dict, Any
try:
    from htrc_features import FeatureReader
    HTRC_AVAILABLE = True
except ImportError:
    HTRC_AVAILABLE = False


class HTRCConnector:
    """Interface to HathiTrust Research Center resources."""

    HTRC_EF_API = "https://data.htrc.illinois.edu/htrc-ef-api/v2"
    WORKSET_API = "https://analytics.hathitrust.org/api/worksets"
    CATALOG_API = "https://catalog.hathitrust.org/api/volumes/brief"

    def __init__(self, token: str = None):
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    def get_workset(self, workset_id: str) -> List[Dict[str, Any]]:
        """Retrieve volume HTIDs from an HTRC Workset."""
        try:
            resp = requests.get(f"{self.WORKSET_API}/{workset_id}", headers=self.headers, timeout=30)
            resp.raise_for_status()
            return [{"htid": h} for h in resp.json().get("htids", [])]
        except Exception as e:
            print(f"Workset fetch failed: {e}")
            return []

    def get_extracted_features(self, htid: str) -> Dict[str, Any]:
        """
        Get EF (Extracted Features) for a volume.
        Inside a Data Capsule, FeatureReader reads from local EF dataset.
        Outside, fetches from HTRC EF API (limited access).
        """
        if HTRC_AVAILABLE:
            try:
                fr = FeatureReader(paths=[htid])
                vol = next(fr.volumes())
                return {
                    "htid": htid,
                    "metadata": vol.metadata,
                    "pages": [
                        {"seq": p.seq, "token_count": p.tokencount,
                         "tokens": p.tokenlist(pos=True, case=False).to_dict()}
                        for p in vol.pages()
                    ]
                }
            except Exception as e:
                print(f"FeatureReader error for {htid}: {e}")
        try:
            resp = requests.get(f"{self.HTRC_EF_API}/ef/{htid}", headers=self.headers, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"EF API error for {htid}: {e}")
        return {}

    def search_accounting_texts(self, max_results: int = 500) -> List[Dict[str, Any]]:
        """Search HathiTrust catalog for accounting-related volumes by subject heading."""
        subjects = [
            "Accounting", "Auditing", "Bookkeeping",
            "Cost accounting", "Financial statements", "Managerial accounting"
        ]
        results = []
        for subject in subjects:
            try:
                resp = requests.get(
                    f"{self.CATALOG_API}/json",
                    params={"subject": subject, "rows": 100},
                    timeout=30
                )
                if resp.ok:
                    for item in resp.json().get("items", [])[:50]:
                        results.append({
                            "htid": item.get("htid"),
                            "title": item.get("title"),
                            "year": item.get("pubdate"),
                            "subject": subject
                        })
            except Exception as e:
                print(f"Catalog search failed for '{subject}': {e}")
            if len(results) >= max_results:
                break
        return results[:max_results]

    def check_rights(self, htid: str) -> str:
        """Return copyright status: 'pd' (public domain) or 'ic' (in-copyright)."""
        try:
            resp = requests.get(f"{self.CATALOG_API}/htid/{htid}.json", timeout=15)
            if resp.ok:
                rights = resp.json().get("items", [{}])[0].get("usRightsString", "")
                return "pd" if "pd" in rights.lower() else "ic"
        except Exception:
            pass
        return "unknown"
