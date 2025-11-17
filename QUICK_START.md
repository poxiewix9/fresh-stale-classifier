# ⚡ Quick Start - 3 Steps!

## What You Need
- Your `best_model.h5` file
- A GitHub account (free)
- 5 minutes

---

## Step 1: Put Your Model File Here
```bash
# Copy your model to this folder
cp /path/to/your/best_model.h5 fresh-stale-api/
```

---

## Step 2: Deploy (Choose ONE)

### 🚂 Railway (Recommended - Easiest)
1. Go to https://railway.app
2. Click "Start a New Project" → "Deploy from GitHub repo"
3. Connect GitHub → Select your repo (or create one)
4. Railway does everything automatically!
5. Copy the URL it gives you

### 🎨 Render (Also Easy)
1. Go to https://render.com
2. Click "New" → "Web Service"
3. Connect GitHub → Select your repo
4. Click "Create Web Service"
5. Copy the URL it gives you

---

## Step 3: Connect to Your App
```bash
cd sustainability
supabase secrets set MODEL_API_URL=https://your-url-here
supabase functions deploy classify-fresh-stale
```

**That's it!** Your app now uses your real model! 🎉

---

## Need Help?

**Don't have GitHub?**
- Create a free account at github.com (takes 1 minute)

**Don't have the model file?**
- It should be in your training folder as `best_model.h5`
- Or check your `fresh-stale-classifier` repo

**Still confused?**
- Railway has a visual tutorial: https://docs.railway.app/getting-started
- Or just tell me what step you're stuck on!

