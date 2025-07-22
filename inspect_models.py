#!/usr/bin/env python3
"""
Script to inspect the actual contents of model directories
"""

import modal
import os

# Define persistent storage volume for model weights
model_volume = modal.Volume.from_name("llm-models-vol", create_if_missing=True)

# Define the container image
app_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2.0.0",
    )
)

app = modal.App("inspect-models", image=app_image)

@app.function(volumes={"/models": model_volume}, timeout=300)
def inspect_models():
    """Inspect the actual contents of model directories"""
    import os
    
    print("🔍 Inspecting model directories...")
    print(f"Current working directory: {os.getcwd()}")
    
    if not os.path.exists("/models"):
        print("❌ /models directory does not exist")
        return
    
    contents = os.listdir("/models")
    print(f"Contents of /models: {contents}")
    
    for item in contents:
        if item == "test.txt":
            continue  # Skip our test file
            
        item_path = f"/models/{item}"
        if os.path.isdir(item_path):
            print(f"\n📁 Directory: {item}")
            try:
                sub_contents = os.listdir(item_path)
                print(f"  Subdirectories: {sub_contents}")
                
                # Inspect each subdirectory
                for sub_item in sub_contents:
                    sub_path = f"{item_path}/{sub_item}"
                    if os.path.isdir(sub_path):
                        print(f"    📂 {sub_item}/")
                        try:
                            files = os.listdir(sub_path)
                            print(f"      Files: {files[:10]}...")  # Show first 10 files
                            
                            # Check for specific model files
                            model_files = [f for f in files if f.endswith(('.safetensors', '.bin', '.pt', '.json', '.txt'))]
                            if model_files:
                                print(f"      Model files: {model_files[:5]}...")
                            
                        except Exception as e:
                            print(f"      Error listing contents: {e}")
                    else:
                        print(f"    📄 {sub_item}")
                        
            except Exception as e:
                print(f"  Error listing contents: {e}")
        else:
            print(f"📄 File: {item}")

if __name__ == "__main__":
    inspect_models.remote() 