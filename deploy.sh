#!/bin/bash

# Simple deployment script for Fresh/Stale Classifier API
# This script helps you deploy to Railway with minimal steps

echo "🚀 Fresh/Stale Classifier API Deployment Helper"
echo "================================================"
echo ""

# Check if model file exists
if [ ! -f "best_model.h5" ]; then
    echo "❌ Error: best_model.h5 not found in current directory"
    echo ""
    echo "Please copy your trained model file here:"
    echo "  cp /path/to/your/best_model.h5 ."
    exit 1
fi

echo "✅ Found best_model.h5"
echo ""

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "📦 Installing Railway CLI..."
    echo ""
    echo "Please install Railway CLI first:"
    echo "  npm install -g @railway/cli"
    echo ""
    echo "Or visit: https://railway.app and deploy via web interface"
    exit 1
fi

echo "✅ Railway CLI found"
echo ""

# Check if logged in
if ! railway whoami &> /dev/null; then
    echo "🔐 Please login to Railway:"
    railway login
fi

echo ""
echo "📤 Deploying to Railway..."
echo ""

# Deploy
railway up

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Copy your Railway URL (shown above)"
echo "2. Run this command in your sustainability folder:"
echo "   cd ../sustainability"
echo "   supabase secrets set MODEL_API_URL=<your-railway-url>"
echo "   supabase functions deploy classify-fresh-stale"
echo ""

