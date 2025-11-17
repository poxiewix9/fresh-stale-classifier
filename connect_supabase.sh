#!/bin/bash

# Script to connect deployed API to Supabase
# Run this after you have your Railway URL

echo "🔗 Connect to Supabase"
echo "======================"
echo ""

if [ -z "$1" ]; then
    echo "Usage: ./connect_supabase.sh <your-railway-url>"
    echo "Example: ./connect_supabase.sh https://your-project.railway.app"
    exit 1
fi

RAILWAY_URL=$1

# Remove trailing slash if present
RAILWAY_URL=${RAILWAY_URL%/}

echo "Setting MODEL_API_URL to: $RAILWAY_URL"
echo ""

# Check if we're in sustainability directory or need to go there
if [ -d "../sustainability" ]; then
    cd ../sustainability
    echo "✅ Changed to sustainability directory"
else
    echo "⚠️  Warning: sustainability directory not found"
    echo "Make sure you're running this from the right location"
fi

echo ""
echo "🔐 Setting Supabase secret..."
supabase secrets set MODEL_API_URL="$RAILWAY_URL"

if [ $? -eq 0 ]; then
    echo "✅ Secret set successfully!"
else
    echo "❌ Failed to set secret. Make sure you're logged into Supabase CLI"
    echo "Run: supabase login"
    exit 1
fi

echo ""
echo "🚀 Deploying Edge Function..."
supabase functions deploy classify-fresh-stale

if [ $? -eq 0 ]; then
    echo ""
    echo "✅✅✅ All done! Your API is now connected!"
    echo ""
    echo "Test it by taking a photo in your app!"
else
    echo "❌ Failed to deploy function"
    exit 1
fi

