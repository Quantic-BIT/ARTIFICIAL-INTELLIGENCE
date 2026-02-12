#!/bin/bash
# ============================================================
#  Policy Assistant RAG – Server Setup Script
#  Run as root on Ubuntu 24.04 VPS: bash setup-server.sh
# ============================================================

set -e

APP_USER="quantic"
APP_DIR="/home/$APP_USER/policy-assistant"
REPO_URL="https://github.com/Quantic-BIT/ARTIFICIAL-INTELLIGENCE.git"
SERVER_IP="138.201.153.167"
PORT=5000

echo "══════════════════════════════════════════════════"
echo "  Policy Assistant RAG – Server Setup"
echo "══════════════════════════════════════════════════"

# ── 1. System packages ──
echo "[1/7] Installing system packages..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv git nginx ufw

# ── 2. Create app user if needed ──
echo "[2/7] Setting up user '$APP_USER'..."
if ! id "$APP_USER" &>/dev/null; then
    useradd -m -s /bin/bash "$APP_USER"
    echo "Created user $APP_USER"
fi

# ── 3. Clone / update repo ──
echo "[3/7] Cloning repository..."
if [ -d "$APP_DIR" ]; then
    echo "Directory exists, pulling latest..."
    cd "$APP_DIR"
    git pull origin main || git pull origin master
else
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

# ── 4. Python venv & dependencies ──
echo "[4/7] Setting up Python environment..."
sudo -u "$APP_USER" bash -c "
    cd $APP_DIR
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    echo '✅ Dependencies installed'
"

# ── 5. Create .env file (user must fill in GROQ_API_KEY) ──
echo "[5/7] Creating .env file..."
ENV_FILE="$APP_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" << 'EOF'
GROQ_API_KEY=your_groq_api_key_here
FLASK_DEBUG=0
PORT=5000
LLM_MODEL=llama-3.1-8b-instant
EOF
    chown "$APP_USER:$APP_USER" "$ENV_FILE"
    echo "⚠️  IMPORTANT: Edit $ENV_FILE and add your GROQ_API_KEY"
else
    echo ".env already exists, skipping"
fi

# ── 6. Create systemd service ──
echo "[6/7] Creating systemd service..."
cat > /etc/systemd/system/policy-assistant.service << EOF
[Unit]
Description=Policy Assistant RAG Application
After=network.target

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$APP_DIR
Environment=PATH=$APP_DIR/venv/bin:/usr/bin
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/venv/bin/gunicorn run:app --bind 127.0.0.1:$PORT --timeout 120 --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable policy-assistant
echo "✅ Systemd service created"

# ── 7. Configure Nginx reverse proxy ──
echo "[7/7] Configuring Nginx..."
cat > /etc/nginx/sites-available/policy-assistant << EOF
server {
    listen 80;
    server_name $SERVER_IP;

    location / {
        proxy_pass http://127.0.0.1:$PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 120s;
        proxy_connect_timeout 120s;
    }
}
EOF

ln -sf /etc/nginx/sites-available/policy-assistant /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl restart nginx

# ── Firewall ──
ufw allow 22/tcp
ufw allow 80/tcp
ufw --force enable

echo ""
echo "══════════════════════════════════════════════════"
echo "  ✅ Setup Complete!"
echo "══════════════════════════════════════════════════"
echo ""
echo "  Next steps:"
echo "  1. Edit the .env file with your GROQ API key:"
echo "     nano $APP_DIR/.env"
echo ""
echo "  2. Start the application:"
echo "     systemctl start policy-assistant"
echo ""
echo "  3. Check status:"
echo "     systemctl status policy-assistant"
echo ""
echo "  4. View logs:"
echo "     journalctl -u policy-assistant -f"
echo ""
echo "  5. Access the app:"
echo "     http://$SERVER_IP"
echo ""
