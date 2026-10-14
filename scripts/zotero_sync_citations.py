"""Pull a Zotero reference set or push the project's verified BibTeX into Zotero (pyzotero).

Credentials come only from environment variables: ZOTERO_LIBRARY_ID, ZOTERO_API_KEY,
ZOTERO_LIBRARY_TYPE (default "user"). They are sent only to api.zotero.org.

pull : export a collection or tag subset to BibTeX + JSON.
       Web API, or --local for a running Zotero 7+ desktop with local API enabled (read-only).
push : plan (default) or --apply creating a project collection/tag, adding missing items from
       the verified BibTeX, filing items that already exist, and filling *empty* fields only.

Examples:
  py -3.9 scripts/zotero_sync_citations.py pull --collection "MI cross"
  py -3.9 scripts/zotero_sync_citations.py push                 # dry-run plan
  py -3.9 scripts/zotero_sync_citations.py push --apply         # writes to Zotero
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

SKILL = Path.home() / ".claude/skills/citation-management/scripts"
sys.path.insert(0, str(SKILL))
from _common import parse_bibtex_file  # noqa: E402

OUT = Path("data/processed/citations_20261001")
FILL_FIELDS = ("DOI", "volume", "issue", "pages", "publicationTitle")


def client(local=False):
    from pyzotero import zotero
    lib = os.environ.get("ZOTERO_LIBRARY_ID")
    if local:
        return zotero.Zotero(lib or "0", os.environ.get("ZOTERO_LIBRARY_TYPE", "user"), local=True)
    key = os.environ.get("ZOTERO_API_KEY")
    if not (lib and key):
        return None
    return zotero.Zotero(lib, os.environ.get("ZOTERO_LIBRARY_TYPE", "user"), key)


def norm_doi(value):
    value = (value or "").strip().lower()
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", value)


def find_collection(zot, name):
    for col in zot.everything(zot.collections()):
        if col["data"]["name"] == name:
            return col["key"]
    return None


def pull(args):
    zot = client(args.local)
    if zot is None:
        sys.exit("No credentials. Set ZOTERO_LIBRARY_ID and ZOTERO_API_KEY, or use --local with Zotero running.")
    if args.collection:
        ckey = find_collection(zot, args.collection)
        if not ckey:
            sys.exit(f"Collection not found: {args.collection}")
        items = zot.everything(zot.collection_items_top(ckey))
        bib = zot.everything(zot.collection_items_top(ckey, content="bibtex"))
        label = re.sub(r"[^A-Za-z0-9]+", "_", args.collection)
    else:
        items = zot.everything(zot.items(tag=args.tag))
        bib = zot.everything(zot.items(tag=args.tag, content="bibtex"))
        label = "tag_" + re.sub(r"[^A-Za-z0-9]+", "_", args.tag)
    items = [i for i in items if i["data"]["itemType"] not in ("attachment", "note", "annotation")]
    (OUT / f"zotero_pull_{label}.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    text = "\n".join(b if isinstance(b, str) else str(b) for b in bib)
    (OUT / f"zotero_pull_{label}.bib").write_text(text, encoding="utf-8")
    print(f"pulled {len(items)} items -> zotero_pull_{label}.bib/.json")


def creators(author_field):
    out = []
    for name in [a.strip() for a in (author_field or "").split(" and ") if a.strip()]:
        if name.lower() == "others":
            continue
        if "," in name:
            last, first = [p.strip() for p in name.split(",", 1)]
        else:
            parts = name.split()
            last, first = parts[-1], " ".join(parts[:-1])
        out.append({"creatorType": "author", "lastName": last.strip("{}"), "firstName": first.strip("{}")})
    return out


def to_zotero(entry, collection_key, tag):
    f = entry["fields"]
    howpublished = f.get("howpublished", "")
    eprint = f.get("eprint") or (howpublished[6:] if howpublished.lower().startswith("arxiv:") else "")
    if entry["type"] == "inproceedings":
        t, venue = "conferencePaper", ("proceedingsTitle", f.get("booktitle", ""))
    elif entry["type"] in ("incollection", "inbook"):
        t, venue = "bookSection", ("bookTitle", f.get("booktitle", ""))
    elif entry["type"] == "misc" and (eprint or "arxiv" in (f.get("journal", "") + f.get("publisher", "")).lower()):
        t, venue = "preprint", ("repository", "arXiv")
    elif entry["type"] == "misc":
        repo = howpublished.replace(" preprint", "") or f.get("publisher", "")
        t, venue = "preprint", ("repository", repo)
    else:
        t, venue = "journalArticle", ("publicationTitle", f.get("journal", ""))
    item = {"itemType": t, "title": f.get("title", "").replace("{", "").replace("}", ""),
            "creators": creators(f.get("author")), "date": f.get("year", ""), venue[0]: venue[1],
            "DOI": f.get("doi", ""), "url": f.get("url", ""), "tags": [{"tag": tag}],
            "collections": [collection_key] if collection_key else []}
    if t == "journalArticle":
        item.update({"volume": f.get("volume", ""), "issue": f.get("number", ""), "pages": f.get("pages", "")})
    if t == "conferencePaper":
        item["pages"] = f.get("pages", "")
    if t == "preprint" and eprint:
        item["archiveID"] = f"arXiv:{eprint}"
    return {k: v for k, v in item.items() if v not in ("", None)}


def push(args):
    entries = parse_bibtex_file(str(args.bib))
    zot = client()
    online = zot is not None
    plan = {"mode": "apply" if args.apply else "dry-run", "online": online, "collection": args.collection,
            "tag": args.tag, "create": [], "file_existing": [], "fill_fields": []}

    existing = {}
    if online:
        for it in zot.everything(zot.items()):
            d = norm_doi(it["data"].get("DOI"))
            if d:
                existing[d] = it
    else:
        cross = json.loads((OUT / "zotero_crossmatch.json").read_text(encoding="utf-8"))
        for ident, hits in cross["found"].items():
            existing[norm_doi(ident)] = {"offline": hits}

    for e in entries:
        doi = norm_doi(e["fields"].get("doi"))
        if doi and doi in existing:
            it = existing[doi]
            plan["file_existing"].append({"doi": doi, "key": it.get("key") or it.get("offline")})
            if online:
                fills = {}
                for zf, bf in zip(FILL_FIELDS, ("doi", "volume", "number", "pages", "journal")):
                    if zf in it["data"] and not it["data"].get(zf) and e["fields"].get(bf):
                        fills[zf] = e["fields"][bf]
                if fills:
                    plan["fill_fields"].append({"key": it["key"], "doi": doi, "fill": fills})
        else:
            plan["create"].append({"citekey": e["key"], "doi": doi, "title": e["fields"].get("title", "")[:90]})

    (OUT / "zotero_push_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"plan: create {len(plan['create'])}, file existing {len(plan['file_existing'])}, "
          f"fill-empty-fields {len(plan['fill_fields'])} -> zotero_push_plan.json (online={online})")
    if not args.apply:
        return
    if not online:
        sys.exit("--apply needs ZOTERO_LIBRARY_ID and ZOTERO_API_KEY with write access.")

    ckey = find_collection(zot, args.collection)
    if not ckey:
        ckey = list(zot.create_collections([{"name": args.collection}])["successful"].values())[0]["key"]
    by_key = {e["key"]: e for e in entries}
    payload = [to_zotero(by_key[c["citekey"]], ckey, args.tag) for c in plan["create"]]
    created, failed = 0, []
    for i in range(0, len(payload), 50):
        resp = zot.create_items(payload[i:i + 50])
        created += len(resp.get("successful", {}))
        failed += list(resp.get("failed", {}).values())
    for f in plan["file_existing"]:
        if not isinstance(f["key"], str):
            continue
        item = zot.item(f["key"])
        zot.addto_collection(ckey, item)
        zot.add_tags(zot.item(f["key"]), args.tag)
    for f in plan["fill_fields"]:
        item = zot.item(f["key"])
        item["data"].update(f["fill"])
        zot.update_item(item)
    log = {"created": created, "failed": failed, "filed": len(plan["file_existing"]), "filled": len(plan["fill_fields"])}
    (OUT / "zotero_push_result.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(log)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--collection")
    g.add_argument("--tag")
    p.add_argument("--local", action="store_true")
    q = sub.add_parser("push")
    q.add_argument("--bib", type=Path, default=OUT / "references_final.bib")
    q.add_argument("--collection", default="MI self paced")
    q.add_argument("--tag", default="MI_selfpaced")
    q.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    pull(a) if a.cmd == "pull" else push(a)


if __name__ == "__main__":
    main()
