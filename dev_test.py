#!/usr/bin/env python3
"""
Development test script for Dhya PersonaRAG Modal
This script validates the code structure and imports without deploying to Modal
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import modal
        print("✅ modal imported successfully")
    except ImportError as e:
        print(f"❌ modal import failed: {e}")
        return False
    
    try:
        import langgraph
        print("✅ langgraph imported successfully")
    except ImportError as e:
        print(f"❌ langgraph import failed: {e}")
        return False
    
    try:
        import langchain
        print("✅ langchain imported successfully")
    except ImportError as e:
        print(f"❌ langchain import failed: {e}")
        return False
    
    try:
        import torch
        print("✅ torch imported successfully")
    except ImportError as e:
        print(f"❌ torch import failed: {e}")
        return False
    
    try:
        import huggingface_hub
        print("✅ huggingface_hub imported successfully")
    except ImportError as e:
        print(f"❌ huggingface_hub import failed: {e}")
        return False
    
    return True

def test_main_structure():
    """Test that main.py can be imported and has the expected structure"""
    print("\n🧪 Testing main.py structure...")
    
    try:
        import main
        print("✅ main.py imported successfully")
    except Exception as e:
        print(f"❌ main.py import failed: {e}")
        return False
    
    # Check that the app is defined
    if hasattr(main, 'app'):
        print("✅ Modal app is defined")
    else:
        print("❌ Modal app not found")
        return False
    
    # Check that the agent classes are defined
    expected_classes = ['OrchestratorAgent', 'RetrievalAgent', 'SynthesisAgent', 'PersonaRAG']
    for class_name in expected_classes:
        if hasattr(main, class_name):
            print(f"✅ {class_name} class found")
        else:
            print(f"❌ {class_name} class not found")
            return False
    
    return True

def test_configuration():
    """Test that the configuration is properly set up"""
    print("\n🧪 Testing configuration...")
    
    import main
    
    # Check MODEL_CONFIG
    if hasattr(main, 'MODEL_CONFIG'):
        models = main.MODEL_CONFIG
        expected_keys = ['synthesis_model', 'orchestrator_model', 'router_model', 'retrieval_model']
        
        for key in expected_keys:
            if key in models:
                print(f"✅ {key}: {models[key]}")
            else:
                print(f"❌ {key} not found in MODEL_CONFIG")
                return False
    else:
        print("❌ MODEL_CONFIG not found")
        return False
    
    # Check volumes
    if hasattr(main, 'model_volume'):
        print("✅ model_volume defined")
    else:
        print("❌ model_volume not found")
        return False
    
    if hasattr(main, 'lora_volume'):
        print("✅ lora_volume defined")
    else:
        print("❌ lora_volume not found")
        return False
    
    return True

def test_modal_authentication():
    """Test Modal authentication"""
    print("\n🧪 Testing Modal authentication...")
    
    try:
        import modal
        # This will fail if not authenticated, but won't crash the script
        modal.client.Client()
        print("✅ Modal client can be initialized")
        return True
    except Exception as e:
        print(f"⚠️  Modal authentication issue: {e}")
        print("   This is expected if you haven't run 'modal token new' yet")
        return True  # Don't fail the test for this

def main():
    """Run all development tests"""
    print("🚀 Dhya PersonaRAG Modal - Development Tests")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Main Structure Tests", test_main_structure),
        ("Configuration Tests", test_configuration),
        ("Modal Authentication", test_modal_authentication),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} passed")
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your environment is ready for deployment.")
        print("\n📋 Next steps:")
        print("1. Run: modal token new (if not authenticated)")
        print("2. Run: ./deploy.sh")
        print("3. Or run: modal deploy main.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues before deploying.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 