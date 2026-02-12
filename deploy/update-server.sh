#!/bin/bash
# ============================================================
#  Policy Assistant RAG – Update Script
#  Run as root: bash update-server.sh
# ============================================================

set -e

APP_USER="quantic"
APP_DIR="/home/$APP_USER/policy-assistant"

echo "Pulling latest code..."
cd "$APP_DIR"
sudo -u "$APP_USER" git pull origin main || sudo -u "$APP_USER" git pull origin master

echo "Updating dependencies..."
sudo -u "$APP_USER" bash -c "
    cd $APP_DIR
    source venv/bin/activate
    pip install -r requirements.txt -q
"

echo "Restarting application..."
systemctl restart policy-assistant

echo "✅ Update complete!"
systemctl status policy-assistant --no-pager
