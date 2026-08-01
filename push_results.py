import os
from huggingface_hub import HfApi
import config


def upload_results(file1_path: str, file2_path: str, hf_token: str = None):
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is required")

    api = HfApi(token=token)
    for path in [file1_path, file2_path]:
        if not os.path.exists(path):
            continue
        filename = os.path.basename(path)
        print(f"   Uploading {filename}...")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=filename,
            repo_id=config.RESULTS_REPO,
            token=token,
            repo_type="dataset"
        )
    print("✅ Upload complete!")
