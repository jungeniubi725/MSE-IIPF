"""Fetch RAVDESS archives or the original CMU feature releases, never results."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.downloads import download, extract_ravdess, fetch_bytes, parse_sdk_splits

CMU_FILES = {
    "mosi": ["labels/CMU_MOSI_Opinion_Labels.csd", "acoustic/CMU_MOSI_OpenSmile_EB10.csd",
             "language/CMU_MOSI_TimestampedWordVectors.csd", "visual/CMU_MOSI_Visual_Facet_42.csd"],
    "mosei": ["labels/CMU_MOSEI_Labels.csd", "acoustic/CMU_MOSEI_COVAREP.csd",
              "language/CMU_MOSEI_TimestampedWordVectors.csd", "visual/CMU_MOSEI_VisualFacet42.csd"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["ravdess", "mosi", "mosei"])
    parser.add_argument("--out-dir", type=Path, help="Default data/ravdess or data/cmu_<dataset>")
    parser.add_argument("--channel", choices=["both", "speech", "song"], default="both")
    parser.add_argument("--actors", type=int, nargs="+", default=list(range(1, 25)))
    parser.add_argument("--record", default="1188976", help="RAVDESS Zenodo record")
    parser.add_argument("--cmu-base-url", default="http://immortal.multicomp.cs.cmu.edu", help="Official SDK server; override with an authorized mirror if unavailable")
    parser.add_argument("--sdk-revision", default="main", help="CMU SDK branch or pinned commit for official split lists")
    parser.add_argument("--splits-only", action="store_true", help="CMU: fetch split lists when CSD/CSV already exists")
    parser.add_argument("--dry-run", action="store_true", help="Print URLs/selection without downloading")
    args = parser.parse_args()
    out = args.out_dir or Path("data") / ("ravdess" if args.dataset == "ravdess" else "cmu_" + args.dataset)
    if any(a < 1 or a > 24 for a in args.actors):
        parser.error("Actors must be between 1 and 24")
    if args.dataset == "ravdess":
        api = f"https://zenodo.org/api/records/{args.record}"
        channels = ["Speech", "Song"] if args.channel == "both" else [args.channel.title()]
        names = {f"Video_{c}_Actor_{a:02d}.zip" for c in channels for a in args.actors}
        names.update(f"Audio_{c}_Actors_01-24.zip" for c in channels)
        if args.dry_run:
            print(json.dumps({"metadata": api, "requested_archives": sorted(names), "destination": str(out)}, indent=2))
            return 0
        record = json.loads(fetch_bytes(api))
        files = {f["key"]: f for f in record["files"]}
        # Actor 18 has no song release in RAVDESS. Verify availability from metadata.
        missing = names - files.keys()
        allowed_missing = {"Video_Song_Actor_18.zip"}
        if missing - allowed_missing:
            raise ValueError(f"Requested archives absent from official record: {sorted(missing)}")
        log = []
        for name in sorted(names & files.keys()):
            item = files[name]
            target = out / "archives" / name
            print(f"Downloading {name}", flush=True)
            receipt = download(item["links"]["self"], target, item.get("checksum"), item.get("size"))
            receipt["extracted_media"] = extract_ravdess(target, out)
            log.append(receipt)
        payload = {"source": api, "missing_archives": sorted(missing), "files": log}
    else:
        ds = "CMU-" + args.dataset.upper()
        stem = "cmu_" + args.dataset
        split_url = (f"https://raw.githubusercontent.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/{args.sdk_revision}/"
                     f"mmsdk/mmdatasdk/dataset/standard_datasets/CMU_{args.dataset.upper()}/{stem}_std_folds.py")
        urls = [f"{args.cmu_base_url.rstrip('/')}/{ds}/{name}" for name in CMU_FILES[args.dataset]]
        if args.dry_run:
            print(json.dumps({"features": [] if args.splits_only else urls, "splits": split_url, "destination": str(out)}, indent=2))
            return 0
        out.mkdir(parents=True, exist_ok=True)
        source = fetch_bytes(split_url)
        splits = parse_sdk_splits(source.decode("utf-8"))
        (out / "official_splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
        import hashlib
        log = []
        if not args.splits_only:
            for url in urls:
                print(f"Downloading {url}", flush=True)
                log.append(download(url, out / "csd" / url.rsplit("/", 1)[-1]))
        payload = {"split_source": split_url, "split_source_sha256": hashlib.sha256(source).hexdigest(), "files": log}
    (out / "download_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

