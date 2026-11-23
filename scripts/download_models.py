"""Download only requested upstream weights; save their resolved revisions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODELS = {
    "llava": ("hf", "lmms-lab/LLaVA-Video-7B-Qwen2", "llava-video"),
    "whisper": ("hf", "openai/whisper-large-v3", "whisper-large-v3"),
    "roberta": ("hf", "FacebookAI/roberta-base", "roberta-base"),
    "videomae": ("hf", "MCG-NJU/videomae-base", "videomae-base"),
    "emotion2vec": ("modelscope", "iic/emotion2vec_base", "emotion2vec-base"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=[*MODELS, "all"], default=["all"])
    parser.add_argument("--out-dir", type=Path, default=Path("weights"))
    parser.add_argument("--revision", help="Revision/commit for a single selected model")
    parser.add_argument("--llava-repo", default=MODELS["llava"][1], help="Use an author's published fine-tuned repository, if available")
    parser.add_argument("--dry-run", action="store_true", help="Print the download plan without network or weights")
    args = parser.parse_args()
    names = list(MODELS) if "all" in args.models else list(dict.fromkeys(args.models))
    if args.revision and len(names) != 1:
        parser.error("--revision applies to one model; select exactly one")
    plan = []
    for name in names:
        provider, repo, folder = MODELS[name]
        repo = args.llava_repo if name == "llava" else repo
        plan.append({"name": name, "provider": provider, "repo": repo,
                     "destination": str(args.out_dir / folder), "revision": args.revision})
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    from src.utils.artifacts import save_json
    for item in plan:
        print(f"Downloading {item['repo']} -> {item['destination']}", flush=True)
        if item["provider"] == "hf":
            from huggingface_hub import HfApi, snapshot_download
            info = HfApi().model_info(item["repo"], revision=item["revision"] or "main")
            item["resolved_revision"] = info.sha
            snapshot_download(repo_id=item["repo"], revision=info.sha, local_dir=item["destination"],
                              ignore_patterns=["*.h5", "*.msgpack", "*.ot", "training_args.bin", "optimizer*", "trainer_state.json", "*.pt"])
        else:
            from modelscope import snapshot_download
            item["requested_revision"] = item["revision"] or "master"
            location = snapshot_download(item["repo"], revision=item["requested_revision"], local_dir=item["destination"])
            item["downloaded_directory"] = str(location)
            # ModelScope's snapshot metadata is retained locally; no unverified commit is invented.
        save_json(Path(item["destination"]) / "download_manifest.json", item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

