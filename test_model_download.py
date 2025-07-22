#!/usr/bin/env python3
"""
Test script to verify model download process
"""

import modal

# Define the models we'll be using
MODEL_CONFIG = {
    "synthesis_model": "meta-llama/Llama-3.1-8B-Instruct",
    "orchestrator_model": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "router_model": "mistralai/Mistral-7B-Instruct-v0.3",
    "retrieval_model": "Qwen/Qwen1.5-7B-Chat",
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

app = modal.App("test-model-download", image=app_image)

@app.function(volumes={"/models": model_volume}, timeout=3600)
def test_single_model_download():
    """Test downloading a single model to verify the process works"""
    from huggingface_hub import snapshot_download
    import os
    
    print("🧪 Testing single model download...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Contents of /models before download: {os.listdir('/models') if os.path.exists('/models') else 'Directory does not exist'}")
    
    # Test with just one model first
    model_id = "deepseek-ai/deepseek-coder-v2-lite-instruct"
    local_path = f"/models/{model_id}"
    
    print(f"📥 Downloading {model_id}...")
    
    try:
        snapshot_download(repo_id=model_id, local_dir=local_path)
        print(f"✅ Successfully downloaded {model_id} to {local_path}")
        
        # Verify the download
        if os.path.exists(local_path):
            size = sum(os.path.getsize(os.path.join(dirpath, filename))
                      for dirpath, dirnames, filenames in os.walk(local_path)
                      for filename in filenames)
            print(f"📊 Model size: {size / (1024**3):.2f} GB")
            
            # List contents to verify
            contents = os.listdir(local_path)
            print(f"📋 Contents: {contents}")
            return True
        else:
            print(f"❌ Model not found at {local_path}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to download {model_id}: {e}")
        return False
    
    print(f"Final contents of /models: {os.listdir('/models') if os.path.exists('/models') else 'Directory does not exist'}")

if __name__ == "__main__":
    print("🧪 Model Download Test")
    print("=" * 30)
    print("Testing single model download to verify the process works...")
    print()
    
    # Run the test
    success = test_single_model_download.remote()
    
    if success:
        print("\n✅ Test successful! Model download process works.")
        print("You can now run the full download: modal run download_models.py")
    else:
        print("\n❌ Test failed! Check the logs for details.") 