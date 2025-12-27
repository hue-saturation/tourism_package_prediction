from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourismPackage/deployment",        # Local folder containing Dockerfile, app.py, requirements.txt
    repo_id="kesavak/tourism-package-prediction",   # Target HF Space repo
    repo_type="space",                             # Upload as HuggingFace Space (auto-deploys)
    path_in_repo="",                               # Upload to root of space
)
print("✅ Deployment files uploaded to HF Spaces!")
print("🚀 Space will auto-deploy: https://huggingface.co/spaces/kesavak/tourism-package-prediction")
