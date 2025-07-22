#!/bin/bash

# Dhya PersonaRAG Modal Deployment Script
# This script automates the deployment process

set -e  # Exit on any error

echo "🚀 Dhya PersonaRAG Modal Deployment"
echo "=================================="

# Check if Modal CLI is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Installing..."
    pip install modal
else
    echo "✅ Modal CLI found"
fi

# Check if user is authenticated
if ! modal token list &> /dev/null; then
    echo "🔑 Modal authentication required"
    echo "Please authenticate with Modal:"
    modal token new
else
    echo "✅ Modal authentication verified"
fi

# Deploy the application
echo ""
echo "📦 Deploying application..."
echo "⚠️  First deployment will take 20-40 minutes to download models"
echo "   Subsequent deployments will be much faster"
echo ""

# Check if models need to be downloaded first
echo "🔍 Checking if models are already downloaded..."
if modal volume ls | grep -q "llm-models-vol"; then
    echo "✅ Model volume exists"
else
    echo "⚠️  Model volume not found. Creating it..."
fi

echo ""
echo "🧪 Testing model download process..."
echo "   This will download one model to verify the process works"
modal run test_model_download.py

echo ""
echo "📥 Downloading all models (this may take 30-60 minutes)..."
echo "   You can monitor progress in the Modal dashboard"
modal run download_models.py

echo ""
echo "🚀 Deploying main application..."
modal deploy main.py

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "1. Copy the endpoint URL from the output above"
echo "2. Test the API using: python test_api.py"
echo "3. Or test with curl:"
echo "   curl -X POST <YOUR_ENDPOINT_URL> \\"
echo "   -H 'Content-Type: application/json' \\"
echo "   -d '{\"query\": \"What are the current income tax slabs in India?\", \"user_id\": \"test_user\"}'"
echo ""
echo "📚 For more information, see README.md" 