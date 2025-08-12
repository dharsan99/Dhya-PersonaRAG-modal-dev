#!/usr/bin/env python3
"""
Manual model download script for Dhya PersonaRAG Modal
This script downloads all required models to the Modal volume before deployment
"""

import modal

# Define the models we'll be using (aligned with main.py)
MODEL_CONFIG = {
    "synthesis_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "orchestrator_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "router_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "retrieval_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

# Define persistent storage volume for model weights
model_volume = modal.Volume.from_name("llm-models-vol", create_if_missing=True)

# Define the container image
app_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2.0.0",  # Fix NumPy compatibility issue
        "huggingface_hub==0.22.2",
        "hf-transfer==0.1.6",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("model-downloader", image=app_image)

@app.function(volumes={"/models": model_volume}, timeout=3600)  # 1 hour timeout
def download_all_models():
    """Download all models to the persistent volume"""
    from huggingface_hub import snapshot_download
    import os
    
    print("🚀 Starting model downloads...")
    print(f"📁 Models will be saved to: /models")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Contents of /models before download: {os.listdir('/models') if os.path.exists('/models') else 'Directory does not exist'}")
    
    for model_id in MODEL_CONFIG.values():
        print(f"\n📥 Downloading {model_id}...")
        local_path = f"/models/{model_id}"
        
        try:
            snapshot_download(repo_id=model_id, local_dir=local_path)
            print(f"✅ Successfully downloaded {model_id} to {local_path}")
            
            # Verify the download
            if os.path.exists(local_path):
                size = sum(os.path.getsize(os.path.join(dirpath, filename))
                          for dirpath, dirnames, filenames in os.walk(local_path)
                          for filename in filenames)
                print(f"📊 Model size: {size / (1024**3):.2f} GB")
                
                # List some contents to verify
                contents = os.listdir(local_path)
                print(f"📋 Contents: {contents[:5]}...")  # Show first 5 items
            else:
                print(f"❌ Model not found at {local_path}")
                
        except Exception as e:
            print(f"❌ Failed to download {model_id}: {e}")
    
    print("\n🎉 Model download process completed!")
    print(f"\n📋 Final contents of /models: {os.listdir('/models') if os.path.exists('/models') else 'Directory does not exist'}")

if __name__ == "__main__":
    print("🔧 Manual Model Downloader for Dhya PersonaRAG")
    print("=" * 50)
    print("This will download all required models to Modal's persistent volume.")
    print("This process may take 30-60 minutes depending on your internet speed.")
    print()
    
    # Run the download
    download_all_models.remote()
    
    print("\n✅ Model download initiated!")
    print("You can now deploy the main application with: modal deploy main.py") 