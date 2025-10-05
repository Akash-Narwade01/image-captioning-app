from huggingface_hub import snapshot_download

# BLIP base model download karega aur local folder me save karega
snapshot_download(
    repo_id="Salesforce/blip-image-captioning-base",
    local_dir="blip_model"  # yaha save hoga
)
