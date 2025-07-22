#!/usr/bin/env python3
"""
Simple test to verify volume functionality
"""

import modal

# Define persistent storage volume for model weights
model_volume = modal.Volume.from_name("llm-models-vol", create_if_missing=True)

# Define the container image
app_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2.0.0",
    )
)

app = modal.App("test-volume", image=app_image)

@app.function(volumes={"/models": model_volume}, timeout=300)
def test_volume_write():
    """Test writing to the volume"""
    import os
    
    print("🧪 Testing volume write functionality...")
    print(f"Current working directory: {os.getcwd()}")
    
    # Test writing a simple file
    test_file = "/models/test.txt"
    with open(test_file, "w") as f:
        f.write("Hello from Modal volume!")
    
    print(f"✅ Wrote test file: {test_file}")
    
    # Test reading the file
    if os.path.exists(test_file):
        with open(test_file, "r") as f:
            content = f.read()
        print(f"✅ Read test file: {content}")
    else:
        print(f"❌ Test file not found: {test_file}")
    
    # List contents
    print(f"Contents of /models: {os.listdir('/models') if os.path.exists('/models') else 'Directory does not exist'}")
    
    return True

@app.function(volumes={"/models": model_volume}, timeout=300)
def test_volume_read():
    """Test reading from the volume"""
    import os
    
    print("🧪 Testing volume read functionality...")
    
    # List contents
    if os.path.exists("/models"):
        contents = os.listdir("/models")
        print(f"Contents of /models: {contents}")
        
        for item in contents:
            item_path = f"/models/{item}"
            if os.path.isdir(item_path):
                print(f"Directory: {item}")
                try:
                    sub_contents = os.listdir(item_path)
                    print(f"  Contents: {sub_contents[:5]}...")  # Show first 5 items
                except Exception as e:
                    print(f"  Error listing contents: {e}")
            else:
                print(f"File: {item}")
    else:
        print("❌ /models directory does not exist")
    
    return True

@app.function(volumes={"/models": model_volume}, timeout=300)
def run_volume_test():
    """Run the complete volume test"""
    print("🧪 Volume Test")
    print("=" * 20)
    
    # Test writing
    print("\n1. Testing volume write...")
    test_volume_write.remote()
    
    # Test reading
    print("\n2. Testing volume read...")
    test_volume_read.remote()
    
    print("\n✅ Volume test completed!")
    return True

if __name__ == "__main__":
    run_volume_test.remote() 