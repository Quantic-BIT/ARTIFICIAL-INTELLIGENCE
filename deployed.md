# Deployed Application

## Live URL

**http://138.201.153.167** (port 80 via Nginx) or **http://138.201.153.167:5000** (direct)

The application is deployed on a self-hosted Ubuntu 24.04 VPS using:
- **Gunicorn** as the WSGI application server
- **Nginx** as the reverse proxy
- **systemd** for process management
- **GitHub Actions** for CI/CD (auto-deploy on push to main)

## Running Locally

```bash
# 1. Clone the repository
git clone <repo-url>
cd projectOne

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Run the application
python run.py
```

The app will be available at `http://localhost:5000`.

## Health Check

Verify the application is running:

```bash
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "healthy",
  "documents_indexed": 54,
  "model": "llama-3.1-8b-instant"
}
```
