# Railway Deployment Guide

## Prerequisites
- Railway account (https://railway.app)
- GitHub account (for connecting repos)

---

## Step 1: Push Code to GitHub

First, create a GitHub repo and push your code:

```bash
cd /Users/anujaysurana/Desktop/voiceIdentity
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/voiceIdentity.git
git push -u origin main
```

---

## Step 2: Deploy Backend on Railway

1. Go to https://railway.app and click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Choose your `voiceIdentity` repo
4. Railway will detect the Dockerfile automatically

### Set Backend Environment Variables:
In Railway dashboard → Variables, add:

| Variable | Value |
|----------|-------|
| `SUPABASE_URL` | `https://koetaaapmxxgvivtftxe.supabase.co` |
| `SUPABASE_SERVICE_KEY` | Your Supabase service key |
| `HF_TOKEN` | Your Hugging Face token |
| `PORT` | `8000` (Railway sets this automatically) |

5. Click **Deploy**
6. Once deployed, get your backend URL (e.g., `https://voiceidentity-backend.up.railway.app`)

---

## Step 3: Deploy Frontend on Railway

1. In Railway, click **"New Service"** in the same project
2. Select **"Deploy from GitHub repo"** again
3. This time, set the **Root Directory** to `frontend`

### Set Frontend Environment Variables:
| Variable | Value |
|----------|-------|
| `VITE_API_URL` | Your backend Railway URL (e.g., `https://voiceidentity-backend.up.railway.app`) |

4. Click **Deploy**
5. Get your frontend URL (e.g., `https://voiceidentity-frontend.up.railway.app`)

---

## Step 4: Update Backend CORS (if needed)

If you get CORS errors, the backend already allows all origins (`allow_origins=["*"]`).

---

## Step 5: Use the App

1. Open your frontend URL on phone or desktop
2. Click the ⚙️ settings icon
3. Enter:
   - **API URL**: Your backend Railway URL
   - **API Key**: `6941bc6fb28082b006ff28b891b9df34a54ea7e129e3456896b647b2af1981de`
   - **User ID**: `anujay`
4. Start recording! 🎤

---

## Troubleshooting

### Backend not starting
- Check Railway logs for errors
- Make sure all environment variables are set
- The first deploy may take 5-10 minutes (ML models are large)

### WebSocket connection fails
- Make sure you're using `https://` URLs
- Backend URL should NOT have trailing slash
- Check browser console for errors

### Microphone not working
- Railway provides HTTPS by default ✓
- Make sure you allow microphone permission

---

## Local Development

Backend:
```bash
cd /Users/anujaysurana/Desktop/voiceIdentity
python -m app.main
```

Frontend:
```bash
cd /Users/anujaysurana/Desktop/voiceIdentity/frontend
npm run dev
```
