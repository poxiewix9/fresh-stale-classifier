# Deployment Guide

## Quick Start: Deploy to Railway (5 minutes)

### Step 1: Prepare Your Model

1. Make sure you have your `best_model.h5` file from training
2. Copy it to the `fresh-stale-api` folder

### Step 2: Deploy to Railway

1. **Sign up/Login**: Go to [railway.app](https://railway.app) and sign in with GitHub

2. **Create New Project**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Authorize Railway to access your GitHub
   - Select your repository
   - Railway will auto-detect it's a Python project

3. **Configure**:
   - Railway will automatically:
     - Detect `requirements.txt`
     - Install dependencies
     - Run `python app.py` (or you can set custom start command)
   
4. **Upload Model File**:
   - In Railway dashboard, go to your service
   - Click "Variables" tab
   - Or use Railway CLI: `railway upload best_model.h5`
   - Or commit `best_model.h5` to your repo (if it's not too large)

5. **Get Your URL**:
   - Railway will give you a URL like: `https://your-project.railway.app`
   - Copy this URL!

### Step 3: Configure Supabase Edge Function

1. **Set the API URL as a secret**:
   ```bash
   cd sustainability
   supabase secrets set MODEL_API_URL=https://your-project.railway.app
   ```

2. **Deploy the Edge Function** (if not already deployed):
   ```bash
   supabase functions deploy classify-fresh-stale
   ```

3. **Test it**:
   ```bash
   supabase functions invoke classify-fresh-stale --body '{"imageUrl": "https://example.com/test-image.jpg"}'
   ```

## Alternative: Deploy to Render

1. Go to [render.com](https://render.com)
2. Click "New" → "Web Service"
3. Connect your GitHub repo
4. Settings:
   - **Name**: fresh-stale-api
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Add environment variable: `MODEL_PATH=best_model.h5` (if needed)
6. Deploy!

## Testing Your Deployment

Once deployed, test your API:

```bash
curl -X POST "https://your-api.railway.app/classify" \
  -H "Content-Type: application/json" \
  -d '{"imageUrl": "https://example.com/test-image.jpg"}'
```

You should get a response like:
```json
{
  "isFresh": true,
  "confidence": 0.95,
  "model": "fresh-stale-classifier"
}
```

## Troubleshooting

### Model not found
- Make sure `best_model.h5` is in the same directory as `app.py`
- Or set `MODEL_PATH` environment variable to the correct path

### CORS errors
- The API already has CORS enabled for all origins
- If you need to restrict, update `allow_origins` in `app.py`

### Slow responses
- First request will be slower (model loading)
- Consider using a service with persistent storage
- Or pre-warm the model on startup

### Model too large
- If model > 100MB, consider:
  - Using model compression (quantization)
  - Hosting on a service with larger limits
  - Using a CDN for the model file

