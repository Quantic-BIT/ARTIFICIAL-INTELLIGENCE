# Deployed Application

## Live URL

🌐 **[Acme Policy Assistant](https://your-app-name.onrender.com)**

> Replace the URL above with your actual Render deployment URL after deploying.

## Deployment Instructions

1. Push your code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click "New" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Name**: acme-policy-assistant
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn run:app`
6. Add Environment Variable:
   - `GROQ_API_KEY`: Your Groq API key
7. Click "Create Web Service"
8. Wait for deployment to complete
9. Update this file with your deployment URL

## Health Check

Verify the deployment is working:

```bash
curl https://your-app-name.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "documents_indexed": 83,
  "model": "llama-3.1-8b-instant"
}
```
