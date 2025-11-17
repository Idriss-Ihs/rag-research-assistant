# 🧠 RAG Research Assistant  
*A modular Retrieval-Augmented Generation system with multi-backend LLM support.*

---

### 🔍 Overview
This project implements a complete **Retrieval-Augmented Generation (RAG)** pipeline — from document ingestion to semantic retrieval and context-aware answer generation.

It is designed for **research applications**, allowing users to query scientific papers or reports and receive precise, referenced answers drawn directly from the source material.

---

### ⚙️ Key Features
- **End-to-end RAG pipeline**: ingestion → preprocessing → embedding → retrieval → generation  
- **Multi-backend LLM support**: switch between  
  - Google **Gemini** *(free & generous)*  
  - **OpenAI** *(managed API)*  
  - **Ollama** *(local open-source models)*  
- **Efficient retrieval** using **FAISS** vector database  
- **Structured logs & YAML configuration** for reproducibility  
- **Streamlit chat interface** for real-time interaction  
- **Offline-capable** (via Ollama) and easy to extend with new models  

---

### 🧱 Project Structure
    src/
    ├── data/ # ingestion, preprocessing, embeddings
    ├── features/ # retrieval, generation, model integration
    ├── app/ # Streamlit interface
    ├── utils/ # logging, config
    data/
    ├── raw/ # source PDFs
    ├── processed/ # cleaned chunks, FAISS index
    └── logs/ # runtime logs

### 🚀 Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
2. **Set up your LLM backend**
3. **Run the Streamlit interface**
    ```bash
    streamlit run src/app/rag_chatbot.py
### 🧩 Configuration

    All paths and settings live in src/config/settings.yaml:
    ```bash
        rag:
        backend: gemini      # options: gemini | openai | ollama
        llm_model: gemini-1.5-flash
        embedding_model: sentence-transformers/all-MiniLM-L6-v2
        top_k: 5
        chunk_size: 500
        temperature: 0.3
    Switch backend or model here — no code changes needed.

<p align="center">
  <img src="assets\rag_answer.png" width="800" alt="CityPulse Dashboard Preview">
  <br>
  <em>Example of rag answer</em>
</p>

<p align="center">
  <img src="assets/original_paper.png" width="800" alt="CityPulse Dashboard Preview">
  <br>
  <em>The paper used to answer the question</em>
</p>