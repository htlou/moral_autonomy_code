#!/usr/bin/env python3
"""Prepare the file layout expected by the original loaders; no model calls."""
from __future__ import annotations
import argparse
import hashlib
import gzip
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "math": ("EleutherAI/hendrycks_math", "21a5633873b6a120296cce3e2df9d5550074f4a3", "math/hendrycks_math", ["*/*.parquet", "README.md"]),
    "mmlu": ("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "general_reasoning/mmlu", ["*/*.parquet", "README.md"]),
    "beavertails": ("PKU-Alignment/BeaverTails", "8401fe609d288129cc684a9b3be6a93e41cfe678", "safety/beavertails", ["round0/330k/*.jsonl.xz", "round0/30k/*.jsonl.gz", "*_30k.jsonl", "README.md"]),
    "bbq": ("https://github.com/nyu-mll/BBQ.git", "bea11bd97d79217245b5871acd247b9d6eb24598", "value/bbq_github", ["data/*.jsonl", "LICENSE", "README.md"]),
}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=SOURCES, default=list(SOURCES))
    p.add_argument("--data-root", type=Path, default=ROOT / "dataset")
    p.add_argument("--source-root", type=Path, help="Copy from an existing dataset/ tree instead of downloading")
    args = p.parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in args.datasets:
        repo, revision, relative, patterns = SOURCES[name]
        dest = args.data_root / relative
        if args.source_root:
            source = args.source_root / relative
            for pattern in patterns:
                for f in sorted(source.glob(pattern)):
                    target = dest / f.relative_to(source)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if f.resolve() != target.resolve():
                        shutil.copy2(f, target)
        elif name == "bbq":
            dest.mkdir(parents=True, exist_ok=True)
            if not (dest / ".git").exists():
                subprocess.run(["git", "init", str(dest)], check=True)
            subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", repo, revision], check=True)
            subprocess.run(["git", "-C", str(dest), "checkout", "--detach", "FETCH_HEAD"], check=True)
        else:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=repo, repo_type="dataset", revision=revision,
                              local_dir=str(dest), allow_patterns=patterns)
        if name == "beavertails":
            for split in ("train", "test"):
                target = dest / f"{split}_30k.jsonl"
                if not target.exists():
                    with gzip.open(dest / "round0/30k" / f"{split}.jsonl.gz", "rb") as source:
                        with target.open("wb") as output:
                            shutil.copyfileobj(source, output)
        data_files = sorted({f for pattern in patterns[:-1] for f in dest.glob(pattern) if f.is_file()})
        if not data_files:
            raise SystemExit(f"No data files found for {name}: {dest}")
        manifest[name] = {"source": repo, "revision": revision,
                          "mode": "local-copy" if args.source_root else "download",
                          "files": {str(f.relative_to(args.data_root)): hashlib.sha256(f.read_bytes()).hexdigest()
                                    for f in data_files}}
        print(f"{name}: {len(data_files)} files prepared")
    out = args.data_root / "download_manifest.json"
    previous = json.loads(out.read_text()) if out.exists() else {}
    previous.update(manifest)
    out.write_text(json.dumps(previous, indent=2) + "\n")
    print(f"Provenance: {out}")

if __name__ == "__main__":
    main()
