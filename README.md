# DocuBot — AI Document Q&A Assistant

Upload any PDF and ask questions about it in plain English. Completely free to run — no paid APIs needed.

---

## How it works

```
PDF -> Extract Text -> Split into Chunks -> Embed (HuggingFace) -> Store in ChromaDB
                                                                          |
User Question -> Embed Question -> Find Similar Chunks -> Send to Groq LLaMA 3 -> Answer
```

This is called RAG (Retrieval Augmented Generation).

---

## Tech Stack

| Tool | Purpose |
|---|---|
| LangChain | RAG pipeline orchestration |
| ChromaDB | Vector database — stores and searches embeddings |
| HuggingFace (all-MiniLM-L6-v2) | Embeddings — free, runs locally on your machine |
| Groq + LLaMA 3 | LLM for answering questions — free API |
| Streamlit | Frontend UI |
| Docker | Containerization |

---

## Project Structure

```
docubot/
├── app.py         <- Streamlit frontend, handles UI and user interaction
├── rag.py         <- Builds the QA chain using Groq LLaMA 3
├── ingest.py      <- Loads PDF, splits into chunks, embeds and stores in ChromaDB
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Setup and Run Locally

**Step 1 — Clone the repo**
```bash
git clone https://github.com/Harish19102003/DocuBot-AI-Document-Q-A-Assistant.git
cd docDocuBot-AI-Document-Q-A-Assistantubot
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

The HuggingFace embedding model (all-MiniLM-L6-v2) will download automatically on first run (~90MB) and cache locally after that.

**Step 3 — Get a free Groq API key**

Go to [console.groq.com](https://console.groq.com), sign up for free (no credit card needed), and copy your API key.

**Step 4 — Run the app**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Enter your Groq API key in the sidebar, upload a PDF, and start asking questions.

---

## Run with Docker

```bash
docker build -t docubot .
docker run -p 8501:8501 docubot
```

---

## Deploy on Streamlit Cloud (Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Go to Settings -> Secrets and add:
```
GROQ_API_KEY = "gsk_your_key_here"
```
5. Click Deploy

---

## Example Use Cases

- Upload a textbook and ask exam questions
- Upload a research paper and get a summary
- Upload a job description and ask what skills are needed
- Upload your resume and ask how to improve it

---

## Customization Ideas

- Add support for multiple PDFs at once
- Show the source page number along with each answer
- Save chat history to a database
- Swap LLaMA 3 with a different Groq model like mixtral-8x7b