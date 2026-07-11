
# Corporate PDF RAG Assistant

A fully containerized Retrieval-Augmented Generation (RAG) application that lets you upload PDF documents, indexes their contents locally using semantic embeddings, and uses the Google Gemini API to answer natural language queries against the document text.

---

## 🚀 Architecture Highlights

The application is split into decoupled services orchestrated via Docker Compose on an isolated bridge network.

- **Backend:** FastAPI (Python 3.11-slim) serving the API.
- **Frontend:** Streamlit dashboard with a persistent document panel.
- **Vector Store:** FAISS (in-memory) for similarity search.
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`).
- **Orchestration:** Multi-container `docker-compose` with service health checks to ensure the frontend waits for the backend.

---

## 📁 Project Directory Structure

```text
rag_assistant/
├── .gitignore
├── .env                  # Local secrets (e.g. GEMINI_API_KEY)
├── docker-compose.yml
├── start-dev.bat
├── start-dev.sh
├── README.md
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── config.py
│   ├── gen_ai.py
│   └── utils.py
└── frontend/
    ├── Dockerfile
    ├── requirements.txt
    └── app.py
```

## ⚡ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/amulshinde/rag_assistant.git
cd rag_assistant
```

### 2. Configure environment variables

Create a `.env` file in the project root (this file is excluded from version control):

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 3. Launch the stack

On Windows (CMD / PowerShell):

```powershell
.\\start-dev.bat
```

On Linux / macOS:

```bash
chmod +x start-dev.sh
./start-dev.sh
```

Docker Compose will build the images and start the services. The frontend waits for the backend healthcheck before serving traffic.

## 📊 Application Services

- Frontend dashboard: http://localhost:8501
- Backend API docs (Swagger): http://localhost:8000/docs

---
