#!/bin/bash

# Complete deployment script - does everything possible automatically
# You'll still need to: 1) Push to GitHub, 2) Deploy on Railway, 3) Set Supabase secret

set -e

echo "🚀 Fresh/Stale API - Complete Setup"
echo "===================================="
echo ""

# Check if we're in the right directory
if [ ! -f "app.py" ] || [ ! -f "best_model.h5" ]; then
    echo "❌ Error: Must run from fresh-stale-api directory"
    exit 1
fi

echo "✅ Found app.py and best_model.h5"
echo ""

# Initialize git if not already
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git repository already exists"
fi

# Add all files
echo ""
echo "📝 Adding files to git..."
git add .
echo "✅ Files added"

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "ℹ️  No changes to commit"
else
    echo ""
    echo "💾 Committing changes..."
    git commit -m "Fresh/Stale Classifier API - Ready to deploy" || echo "⚠️  Commit failed (might be empty)"
    echo "✅ Changes committed"
fi

echo ""
echo "===================================="
echo "✅ Local setup complete!"
echo ""
echo "Next steps (you need to do these):"
echo ""
echo "1️⃣  Push to GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Create a new repo (e.g., 'fresh-stale-api')"
echo "   - Then run:"
echo "     git remote add origin https://github.com/YOUR_USERNAME/fresh-stale-api.git"
echo "     git push -u origin main"
echo ""
echo "2️⃣  Deploy to Railway:"
echo "   - Go to https://railway.app"
echo "   - Click 'Start a New Project' → 'Deploy from GitHub repo'"
echo "   - Select your repo"
echo "   - Wait 2-3 minutes, copy the URL"
echo ""
echo "3️⃣  Connect to Supabase:"
echo "   cd ../sustainability"
echo "   supabase secrets set MODEL_API_URL=https://your-railway-url"
echo "   supabase functions deploy classify-fresh-stale"
echo ""
echo "Or run: ./deploy_github.sh (if you want help with GitHub)"
echo ""

