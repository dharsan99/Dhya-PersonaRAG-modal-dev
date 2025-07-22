#!/usr/bin/env python3
"""
Simple test script for the PersonaRAG API
"""

import requests
import time
import json

def test_health_check():
    """Test the health check endpoint"""
    url = "https://dharsan99--dhya-persona-rag-pipeline-personarag-health-check.modal.run"
    
    print("🏥 Testing health check endpoint...")
    print(f"URL: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Health check successful!")
            return True
        else:
            print("❌ Health check failed!")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Health check timed out (30 seconds)")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_main_api():
    """Test the main API endpoint"""
    url = "https://dharsan99--dhya-persona-rag-pipeline-personarag-run.modal.run"
    
    payload = {
        "query": "What are the benefits of investing in mutual funds?",
        "user_id": "test_user_1"
    }
    
    print("\n🚀 Testing main API endpoint...")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("\n⏳ This may take several minutes for the first request as models need to load...")
    
    try:
        # Use a much longer timeout for the first request
        print("📡 Sending request (timeout: 10 minutes)...")
        response = requests.post(url, json=payload, timeout=600)  # 10 minutes
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API call successful!")
            print(f"Response: {json.dumps(result, indent=2)}")
            return True
        elif response.status_code == 204:
            print("⚠️  API returned 204 (No Content)")
            print("This usually means the request was processed but the client disconnected")
            print("The models might still be loading. Try again in a few minutes.")
            return False
        else:
            print(f"❌ API call failed with status {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ API call timed out (10 minutes)")
        print("This is normal for the first request as models need to load")
        print("The models are likely still initializing. Try again in 2-3 minutes.")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"🔌 Connection error: {e}")
        print("The service might be restarting or the models are still loading.")
        return False
    except Exception as e:
        print(f"❌ API call error: {e}")
        return False

def test_with_retry():
    """Test the API with retries"""
    print("\n🔄 Testing with retries...")
    
    for attempt in range(3):
        print(f"\n--- Attempt {attempt + 1}/3 ---")
        
        if test_main_api():
            print("🎉 Success! API is working correctly.")
            return True
        else:
            if attempt < 2:  # Don't wait after the last attempt
                wait_time = (attempt + 1) * 60  # 1, 2, 3 minutes
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
    
    print("❌ All attempts failed. The service might need more time to initialize.")
    return False

def main():
    print("🧪 PersonaRAG API Test")
    print("=" * 30)
    
    # Test health check first
    health_ok = test_health_check()
    
    if health_ok:
        print("\n🎉 Health check passed! Testing main API...")
        test_with_retry()
    else:
        print("\n⚠️  Health check failed. The service might still be initializing.")
        print("This is normal for the first deployment. Try again in a few minutes.")

if __name__ == "__main__":
    main() 