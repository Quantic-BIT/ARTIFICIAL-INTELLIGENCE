# Acme Policy Assistant

A Retrieval-Augmented Generation (RAG) application that answers questions about company policies and procedures using AI.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🚀 Features

- **Intelligent Q&A**: Ask questions about company policies in natural language
- **Source Citations**: Every answer includes citations to the source documents
- **Modern Chat UI**: Beautiful, responsive dark-themed interface
- **Fast Responses**: Powered by Groq's fast LLM inference
- **Guardrails**: Refuses off-topic questions and always cites sources
- **Evaluation Suite**: Comprehensive evaluation framework included

## 📋 Policy Coverage

The system covers these Acme Corporation policies:
- 📅 PTO & Leave Policy
- 🏠 Remote Work Policy
- 💰 Expense Reimbursement
- 🔒 IT Security Policy
- 🎁 Employee Benefits
- 📆 Holiday Schedule
- 📜 Code of Conduct
- 📖 Employee Handbook

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Groq (llama-3.1-8b-instant) |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Vector Store** | ChromaDB |
| **Web Framework** | Flask |
| **Deployment** | Local (Gunicorn-ready) |

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- A Groq API key (free at https://console.groq.com)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/policy-assistant.git
   cd policy-assistant
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

5. **Run the application**
   ```bash
   python run.py
   ```

6. **Open in browser**
   ```
   http://localhost:5000
   ```

## 🔧 Configuration

Create a `.env` file with the following variables:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional
LLM_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=all-MiniLM-L6-v2
FLASK_DEBUG=0
SECRET_KEY=your-secret-key
```

## 📁 Project Structure

```
projectOne/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── main.py               # Routes and endpoints
│   ├── rag/
│   │   ├── ingestion.py      # Document loading & chunking
│   │   ├── embeddings.py     # Embedding model wrapper
│   │   ├── vectorstore.py    # ChromaDB operations
│   │   └── chain.py          # RAG chain with LLM
│   ├── templates/
│   │   └── index.html        # Chat interface
│   └── static/
│       └── style.css         # UI styling
├── policies/                  # Company policy documents
├── evaluation/
│   ├── questions.json        # Test questions
│   └── evaluate.py           # Evaluation script
├── .github/workflows/
│   └── ci-cd.yml             # CI/CD pipeline
├── requirements.txt
├── run.py                    # Entry point
├── README.md
├── deployed.md
├── design-and-evaluation.md
└── ai-use.md
```

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat interface |
| `/chat` | POST | Send message, receive answer |
| `/health` | GET | Health check |
| `/api/reindex` | POST | Force re-index documents |

### Chat API Example

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How many vacation days do I get?"}'
```

Response:
```json
{
  "answer": "New employees (0-2 years) receive 15 vacation days per year...",
  "sources": [
    {"source": "pto_policy.md", "title": "PTO Policy", "snippet": "..."}
  ],
  "latency_ms": 523.4
}
```

## 📊 Evaluation

Run the evaluation suite:

```bash
python evaluation/evaluate.py
```

This evaluates:
- **Groundedness**: Are answers factually supported by retrieved documents?
- **Citation Accuracy**: Do citations point to correct sources?
- **Partial/Exact Match**: Token-level overlap with gold answers
- **Latency**: Response time metrics (p50, p95) over on-topic queries
- **Off-topic Handling**: Are irrelevant questions properly refused?

Latest results: **100% groundedness**, **100% citation accuracy**, **85.4% partial match**, **75.0% exact match**, **100% off-topic handling**.

Save results to JSON:
```bash
python evaluation/evaluate.py --save
```

Run with ablation study (compares k=3, k=5, k=8):
```bash
python evaluation/evaluate.py --save --ablation
```

## 🚀 Running in Production

Use gunicorn for production:

```bash
gunicorn run:app --bind 0.0.0.0:5000 --timeout 120
```

### CI/CD

The project includes GitHub Actions that:
- Run on push/PR to main
- Install dependencies and run import/ingestion checks
- Lint with flake8

## 🧪 Testing

Run import check:
```bash
python -c "from app import create_app; app = create_app()"
```

Run ingestion test:
```bash
python -c "from app.rag.ingestion import ingest_documents; print(len(ingest_documents()))"
```

## 📄 License

MIT License - feel free to use this project for learning and development.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For questions about this project, please open an issue on GitHub.

---

Built with ❤️ for the AI Engineering course project.
# ARTIFICIAL-INTELLIGENCE
