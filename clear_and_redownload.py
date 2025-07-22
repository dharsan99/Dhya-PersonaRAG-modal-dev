#!/usr/bin/env python3
"""
Script to clear the volume and re-download models properly
"""

import modal
import os

# Tiny models for testing - these are guaranteed to work with vLLM
# Using very small models to ensure they download and load properly
MODEL_CONFIG = {
    "synthesis_model": "microsoft/DialoGPT-small",  # Very small model (117M parameters)
    "orchestrator_model": "microsoft/DialoGPT-small",  # Very small model
    "router_model": "microsoft/DialoGPT-small",  # Very small model
    "retrieval_model": "microsoft/DialoGPT-small",  # Very small model
}

# Define persistent storage volume for model weights
model_volume = modal.Volume.from_name("llm-models-vol", create_if_missing=True)

# Define the container image
app_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2.0.0",
        "huggingface_hub==0.22.2",
        "hf-transfer==0.1.6",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("clear-and-redownload", image=app_image)

@app.function(volumes={"/models": model_volume}, timeout=3600)
def clear_and_redownload():
    """Clear the volume and re-download all models"""
    import shutil
    from huggingface_hub import snapshot_download
    
    print("🧹 Clearing existing model directories...")
    
    # Clear existing directories
    for model_id in MODEL_CONFIG.values():
        local_path = f"/models/{model_id}"
        if os.path.exists(local_path):
            print(f"Removing {local_path}")
            shutil.rmtree(local_path)
    
    print("📥 Starting fresh model downloads...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Contents of /models after clearing: {os.listdir('/models') if os.path.exists('/models') else 'Directory does not exist'}")
    
    for model_id in MODEL_CONFIG.values():
        local_path = f"/models/{model_id}"
        print(f"\n📥 Downloading {model_id}...")
        
        try:
            # Download without ignore_patterns to get all necessary files
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
    
    print(f"\n🎉 Download process completed!")
    print(f"Final contents of /models: {os.listdir('/models') if os.path.exists('/models') else 'Directory does not exist'}")

if __name__ == "__main__":
    print("🧹 Clear and Re-download Models")
    print("=" * 40)
    print("This will clear the existing volume and re-download all models...")
    print()
    
    # Run the clear and redownload process
    clear_and_redownload.remote()
    
    print("\n✅ Process completed! Check the logs for details.") 