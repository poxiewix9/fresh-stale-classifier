# Super Simple Deployment Guide (5 Minutes)

## Option 1: Railway (Easiest - No Code!)

### Step 1: Prepare Files
1. Make sure you have `best_model.h5` in the `fresh-stale-api` folder
2. That's it! Railway will handle the rest.

### Step 2: Deploy via Railway Website (No Terminal Needed!)

1. **Go to Railway**: https://railway.app
2. **Sign up** with GitHub (click "Start a New Project")
3. **Click "Deploy from GitHub repo"**
4. **If your code is on GitHub**: Select your repo
   - **If NOT on GitHub yet**: 
     - Create a new GitHub repo
     - Push the `fresh-stale-api` folder to it
     - Then select it in Railway
5. **Railway will automatically**:
   - Detect it's Python
   - Install dependencies
   - Deploy your API
6. **Get your URL**: Railway will show you a URL like `https://your-project.railway.app`
7. **Copy that URL!**

### Step 3: Connect to Supabase (2 commands)

Open terminal in your `sustainability` folder and run:

```bash
supabase secrets set MODEL_API_URL=https://your-project.railway.app
supabase functions deploy classify-fresh-stale
```

**Done!** 🎉

---

## Option 2: Even Simpler - Use Render (Also No Terminal!)

1. **Go to Render**: https://render.com
2. **Sign up** with GitHub
3. **Click "New" → "Web Service"**
4. **Connect your GitHub repo** (or create one and push `fresh-stale-api`)
5. **Settings**:
   - Name: `fresh-stale-api`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
6. **Click "Create Web Service"**
7. **Wait 2-3 minutes** for deployment
8. **Copy your URL** (like `https://fresh-stale-api.onrender.com`)
9. **Run the 2 Supabase commands above** with your Render URL

---

## Option 3: Test Locally First (If You Want)

If you want to test before deploying:

```bash
cd fresh-stale-api
pip install -r requirements.txt
python app.py
```

Then test:
```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"imageUrl": "https://example.com/test.jpg"}'
```

---

## Troubleshooting

**"Model not found" error?**
- Make sure `best_model.h5` is in the same folder as `app.py`
- Or set environment variable `MODEL_PATH=best_model.h5` in Railway/Render

**"Port already in use" error?**
- Change the port in `app.py` or set `PORT` environment variable

**Still stuck?**
- Railway has great docs: https://docs.railway.app
- Render has docs: https://render.com/docs
- Or just ask me! 😊

