import os
import zipfile
import requests
from io import BytesIO

def download_github_repo():
    repo_zip_url = "https://github.com/tech-srl/bottleneck/archive/refs/heads/main.zip"

    # Directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Target directory name
    target_dir = os.path.join(script_dir, "bottleneck-main")

    print("Downloading repository...")
    response = requests.get(repo_zip_url)
    response.raise_for_status()

    print("Extracting repository...")
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(script_dir)

    # GitHub ZIP extracts to "bottleneck-main" by default, but ensure correct naming
    extracted_dir = os.path.join(script_dir, "bottleneck-main")
    if not os.path.exists(extracted_dir):
        # Fallback in case branch name differs
        for name in os.listdir(script_dir):
            if name.startswith("bottleneck-"):
                os.rename(
                    os.path.join(script_dir, name),
                    extracted_dir
                )
                break

    print(f"Repository successfully downloaded to: {target_dir}")

if __name__ == "__main__":
    download_github_repo()
