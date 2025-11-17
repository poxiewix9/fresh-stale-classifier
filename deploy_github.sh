#!/bin/bash

# Helper script to push to GitHub
# Run this after creating your GitHub repo

echo "🔗 GitHub Deployment Helper"
echo "============================"
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Git not initialized. Run deploy_all.sh first"
    exit 1
fi

# Check if remote exists
if git remote get-url origin &>/dev/null; then
    echo "✅ Remote already configured:"
    git remote get-url origin
    echo ""
    read -p "Push to this remote? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📤 Pushing to GitHub..."
        git push -u origin main || git push -u origin master
        echo "✅ Pushed to GitHub!"
    fi
else
    echo "📝 No remote configured yet."
    echo ""
    echo "Steps:"
    echo "1. Go to https://github.com/new"
    echo "2. Create a new repository (e.g., 'fresh-stale-api')"
    echo "3. Copy the repository URL"
    echo ""
    read -p "Enter your GitHub repo URL: " repo_url
    
    if [ -z "$repo_url" ]; then
        echo "❌ No URL provided"
        exit 1
    fi
    
    echo ""
    echo "🔗 Adding remote..."
    git remote add origin "$repo_url"
    echo "✅ Remote added"
    
    echo ""
    echo "📤 Pushing to GitHub..."
    git push -u origin main || git push -u origin master
    echo "✅ Pushed to GitHub!"
    echo ""
    echo "🎉 Now go to Railway and deploy!"
fi

