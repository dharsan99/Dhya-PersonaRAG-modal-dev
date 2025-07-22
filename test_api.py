#!/usr/bin/env python3
"""
Test script for the Dhya PersonaRAG Modal API
Replace the URL with your actual Modal endpoint URL
"""

import requests
import json
import sys

def test_api(url: str, query: str = None, user_id: str = "test_user"):
    """
    Test the PersonaRAG API endpoint
    
    Args:
        url: The Modal endpoint URL
        query: The test query (defaults to a sample query)
        user_id: Test user ID
    """
    
    if query is None:
        query = "What are the current income tax slabs in India for the new regime?"
    
    payload = {
        "query": query,
        "user_id": user_id
    }
    
    print(f"🚀 Testing API endpoint: {url}")
    print(f"📝 Query: {query}")
    print(f"👤 User ID: {user_id}")
    print("-" * 50)
    
    try:
        response = requests.post(url, json=payload, timeout=300)  # 5 minute timeout
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"📄 Response: {result.get('response', 'No response field found')}")
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Timeout: Request took too long (5+ minutes)")
        print("This is normal for the first request as models need to load")
    except requests.exceptions.ConnectionError:
        print("🔌 Connection Error: Could not connect to the endpoint")
        print("Make sure the URL is correct and the app is deployed")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def main():
    # Replace this with your actual Modal endpoint URL
    # You'll get this URL after running: modal deploy main.py
    default_url = "https://dharsan99--dhya-persona-rag-pipeline-personarag-run.modal.run"
    
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input(f"Enter your Modal endpoint URL (or press Enter to use default): ").strip()
        if not url:
            url = default_url
    
    # Test queries
    test_queries = [
        "What are the current income tax slabs in India for the new regime?",
        "Explain the benefits of investing in mutual funds for beginners",
        "What are the key differences between traditional and digital banking?",
    ]
    
    print("🧪 Running API tests...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Test {i}/{len(test_queries)}")
        test_api(url, query, f"test_user_{i}")
        print("\n" + "="*60 + "\n")
        
        if i < len(test_queries):
            input("Press Enter to continue to next test...")

if __name__ == "__main__":
    main() 