#!/usr/bin/env python3
"""
Test script to verify Qwen model is properly downloaded and accessible
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
        "vllm==0.4.0",
        "torch==2.1.2",
    )
)

app = modal.App("test-qwen", image=app_image)

@app.function(volumes={"/models": model_volume}, timeout=300)
def test_qwen_model():
    """Test if Qwen model can be loaded by vLLM"""
    import os
    
    print("🧪 Testing Qwen model loading...")
    print(f"Current working directory: {os.getcwd()}")
    
    model_path = "/models/Qwen/Qwen1.5-7B-Chat"
    
    # Check if model directory exists
    if not os.path.exists(model_path):
        print(f"❌ Model directory not found: {model_path}")
        return False
    
    print(f"✅ Model directory found: {model_path}")
    
    # List contents
    contents = os.listdir(model_path)
    print(f"📋 Contents: {contents}")
    
    # Check for specific model files
    required_files = [
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors", 
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "config.json"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file} exists ({size} bytes)")
        else:
            print(f"❌ {file} missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    # Try to load with vLLM
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        
        print("🚀 Attempting to load model with vLLM...")
        
        engine_args = AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        )
        
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("✅ Successfully loaded Qwen model with vLLM!")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model with vLLM: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Qwen Model Test")
    print("=" * 20)
    
    success = test_qwen_model.remote()
    
    if success:
        print("\n✅ Qwen model test successful!")
        print("The model is ready for use.")
    else:
        print("\n❌ Qwen model test failed!")
        print("Check the logs for details.") 