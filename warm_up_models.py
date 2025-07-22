#!/usr/bin/env python3
"""
Script to warm up the PersonaRAG models
"""

import requests
import time
import json

def warm_up_models():
    """Warm up all models via the API endpoint"""
    url = "https://dharsan99--dhya-persona-rag-pipeline-personarag-warm-up.modal.run"
    
    print("🔥 Warming up PersonaRAG models...")
    print(f"URL: {url}")
    
    try:
        response = requests.post(url, timeout=600)  # 10 minute timeout
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Models warmed up successfully!")
            print(f"Response: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"❌ Warm-up failed: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Warm-up timed out (10 minutes)")
        print("This is normal for the first time as models need to be downloaded and loaded")
        return False
    except Exception as e:
        print(f"❌ Warm-up error: {e}")
        return False

def main():
    print("🚀 PersonaRAG Model Warm-up")
    print("=" * 30)
    
    success = warm_up_models()
    
    if success:
        print("\n🎉 Models are ready! You can now test the API.")
    else:
        print("\n⚠️  Warm-up failed. The models might still be initializing.")
        print("Try again in a few minutes.")

if __name__ == "__main__":
    main() 