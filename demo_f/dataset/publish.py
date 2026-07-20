"""Explicitly publish a validated local release to Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from .contract import DEFAULT_ROOT, REPOSITORY_ID
from .validate import validate_release


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo-id", default=REPOSITORY_ID)
    parser.add_argument("--confirm-public-upload", action="store_true")
    args = parser.parse_args()
    if not args.confirm_public_upload:
        raise SystemExit("refusing external upload without --confirm-public-upload")
    report = validate_release(args.root, require_complete=True)
    api = HfApi()
    identity = api.whoami()
    print(f"authenticated as {identity['name']} | uploading {report['clips']:,} clips")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=args.root,
        commit_message=f"Publish retargeted dataset schema with {report['clips']:,} clips",
    )
    print(f"published https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
