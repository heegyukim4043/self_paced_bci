from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path


BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

QUERIES = {
    "onset": (
        '(("self-paced"[Title/Abstract] OR "self initiated"[Title/Abstract]) '
        'AND (EEG[Title/Abstract] OR electroencephalography[Title/Abstract]) '
        'AND ("movement onset"[Title/Abstract] OR "motor intention"[Title/Abstract]))'
    ),
    "continuous_eval": (
        '((asynchronous[Title/Abstract] OR "self-paced"[Title/Abstract]) '
        'AND "motor imagery"[Title/Abstract] '
        'AND ("false positive"[Title/Abstract] OR "false alarm"[Title/Abstract] '
        'OR idle[Title/Abstract] OR "non-control"[Title/Abstract]))'
    ),
    "feedback": (
        '("motor imagery"[Title/Abstract] '
        'AND (sham[Title/Abstract] OR yoked[Title/Abstract] '
        'OR placebo[Title/Abstract]) '
        'AND (feedback[Title/Abstract] OR neurofeedback[Title/Abstract]))'
    ),
    "session_variability": (
        '("motor imagery"[Title/Abstract] '
        'AND (intersession[Title/Abstract] OR "inter-session"[Title/Abstract] '
        'OR "day-to-day"[Title/Abstract] OR longitudinal[Title/Abstract]) '
        'AND (EEG[Title/Abstract] OR electroencephalography[Title/Abstract]))'
    ),
    "onset_reference": (
        '(("self-initiated"[Title/Abstract] OR "self-paced"[Title/Abstract]) '
        'AND EEG[Title/Abstract] AND onset[Title/Abstract] '
        'AND (EMG[Title/Abstract] OR kinematic*[Title/Abstract] '
        'OR "motion capture"[Title/Abstract] OR accelerometer[Title/Abstract]))'
    ),
    "within_between_day": (
        '("motor imagery"[Title/Abstract] '
        'AND ("same day"[Title/Abstract] OR "within-day"[Title/Abstract] '
        'OR "between-day"[Title/Abstract] OR "across days"[Title/Abstract]) '
        'AND (EEG[Title/Abstract] OR BCI[Title/Abstract]))'
    ),
    "feedback_control": (
        '("motor imagery"[Title/Abstract] '
        'AND (yoked[Title/Abstract] OR playback[Title/Abstract] '
        'OR "sham feedback"[Title/Abstract] OR "false feedback"[Title/Abstract]) '
        'AND (EEG[Title/Abstract] OR BCI[Title/Abstract] '
        'OR neurofeedback[Title/Abstract]))'
    ),
}


def request(endpoint: str, params: dict[str, str | int]) -> bytes:
    safe_params = {
        **params,
        "tool": "mi_selfpaced_direct_collection_review",
    }
    url = f"{BASE}/{endpoint}?{urllib.parse.urlencode(safe_params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "mi-selfpaced-review/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def text_of(node: ET.Element | None) -> str | None:
    if node is None:
        return None
    text = "".join(node.itertext()).strip()
    return text or None


def parse_articles(payload: bytes) -> list[dict[str, object]]:
    root = ET.fromstring(payload)
    records: list[dict[str, object]] = []
    for article in root.findall(".//PubmedArticle"):
        citation = article.find("MedlineCitation")
        if citation is None:
            continue
        journal_article = citation.find("Article")
        if journal_article is None:
            continue
        ids: dict[str, str] = {}
        for item in article.findall("PubmedData/ArticleIdList/ArticleId"):
            if item.text:
                ids[item.attrib.get("IdType", "unknown")] = item.text.strip()
        authors = []
        for author in journal_article.findall(".//Author"):
            collective = text_of(author.find("CollectiveName"))
            if collective:
                authors.append(collective)
                continue
            family = text_of(author.find("LastName")) or ""
            initials = text_of(author.find("Initials")) or ""
            name = " ".join(part for part in (family, initials) if part)
            if name:
                authors.append(name)
        abstract_parts = [
            text_of(item)
            for item in journal_article.findall(".//Abstract/AbstractText")
        ]
        records.append(
            {
                "pmid": text_of(citation.find("PMID")),
                "title": text_of(journal_article.find("ArticleTitle")),
                "abstract": " ".join(part for part in abstract_parts if part) or None,
                "authors": authors,
                "journal": text_of(journal_article.find("Journal/Title")),
                "year": text_of(journal_article.find("Journal/JournalIssue/PubDate/Year")),
                "doi": ids.get("doi"),
                "pmcid": ids.get("pmc"),
                "publication_types": [
                    text_of(item)
                    for item in journal_article.findall("PublicationTypeList/PublicationType")
                    if text_of(item)
                ],
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="data/processed/direct_collection_literature_review_20260924/sources",
    )
    parser.add_argument("--retmax", type=int, default=200)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, (name, query) in enumerate(QUERIES.items()):
        search_payload = request(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": args.retmax,
                "sort": "relevance",
            },
        )
        search = json.loads(search_payload)["esearchresult"]
        ids = search.get("idlist", [])
        time.sleep(0.4)
        records = []
        if ids:
            records = parse_articles(
                request(
                    "efetch.fcgi",
                    {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
                )
            )
        output = {
            "database": "PubMed",
            "access_date": date.today().isoformat(),
            "query": query,
            "expected_total": int(search.get("count", 0)),
            "retrieved_total": len(records),
            "complete": int(search.get("count", 0)) <= args.retmax,
            "pmids": ids,
            "records": records,
            "provenance": {
                "esearch": f"{BASE}/esearch.fcgi",
                "efetch": f"{BASE}/efetch.fcgi",
                "retmax": args.retmax,
                "sort": "relevance",
            },
        }
        path = output_dir / f"pubmed_{name}.json"
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"{name}: total={output['expected_total']} "
            f"retrieved={output['retrieved_total']} -> {path}"
        )
        if index < len(QUERIES) - 1:
            time.sleep(0.4)


if __name__ == "__main__":
    main()
