# 🚀 Deploy Now - Super Simple!

Everything is ready! Your `best_model.h5` is already here. Just follow these steps:

## Option 1: Railway (Easiest - 2 minutes)

1. **Go to**: https://railway.app
2. **Click**: "Start a New Project" → "Deploy from GitHub repo"
3. **If you don't have a GitHub repo yet**:
   - Go to https://github.com/new
   - Create a new repo (e.g., "fresh-stale-api")
   - Push this folder to it:
     ```bash
     cd fresh-stale-api
     git init
     git add .
     git commit -m "Initial commit"
     git remote add origin https://github.com/YOUR_USERNAME/fresh-stale-api.git
     git push -u origin main
     ```
4. **Back in Railway**: Select your repo
5. **Railway will automatically**:
   - Detect Python
   - Install dependencies
   - Deploy!
6. **Copy the URL** Railway gives you (like `https://your-project.railway.app`)

## Option 2: Render (Also Easy)

1. **Go to**: https://render.com
2. **Click**: "New" → "Web Service"
3. **Connect GitHub** and select your repo
4. **Settings**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. **Click**: "Create Web Service"
6. **Wait 2-3 minutes**, then copy your URL

## Connect to Your App

Once you have your URL, run these 2 commands:

```bash
cd ../sustainability
supabase secrets set MODEL_API_URL=https://your-url-here
supabase functions deploy classify-fresh-stale
```

**Done!** 🎉

## Test Locally First (Optional)

If you want to test before deploying:

```bash
cd fresh-stale-api
pip install -r requirements.txt
python app.py
```

Then in another terminal:
```bash
python test_local.py
```

---

## Need Help?

Just tell me:
- Which step you're on
- What error you're seeing
- And I'll help you fix it!

