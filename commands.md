# Commands to Run

## 1. Activate Virtual Environment

```bash
source venv/bin/activate
```

## 2. Run Evaluation

```bash
python evaluation/evaluate.py --save
```

## 3. Run Ablation Study

```bash
python evaluation/evaluate.py --save --ablation
```

## 4. Run Basic Tests

```bash
python -c "from app import create_app; app = create_app(); print('✅ App imports successfully')"
python -c "from app.rag.ingestion import ingest_documents; print(f'✅ Ingestion works: {len(ingest_documents())} chunks')"
```

## 5. Run the App Locally

```bash
PORT=5001 python run.py
```

Then open http://localhost:5001

## 6. Deploy to Server

```bash
ssh root@138.201.153.167
git clone https://github.com/Quantic-BIT/ARTIFICIAL-INTELLIGENCE.git /home/quantic/policy-assistant
bash /home/quantic/policy-assistant/deploy/setup-server.sh
nano /home/quantic/policy-assistant/.env
systemctl start policy-assistant
systemctl status policy-assistant
```

Then open http://138.201.153.167:5000
