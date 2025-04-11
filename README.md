#pdf-summarizer
📚 PDF Summarizer & Q/A Chatbot using LLMs
This Streamlit app allows you to upload any PDF document, summarize it, and interact with its contents using natural language questions — all powered by OpenAI's language models and LangChain.

🚀 Features
PDF Upload: Upload any text-based PDF file.

Smart Summarization: Instantly get a concise summary of the entire document.

Ask Questions: Chat with your PDF and get accurate answers from the content.

Embeddings + Vector Search: Uses OpenAI Embeddings with FAISS for efficient semantic search.

Secure Key Handling: API key is stored safely using Streamlit’s secrets.toml.

🧰 Built With
Streamlit — UI and app framework

LangChain — LLM orchestration and chaining

OpenAI API — for GPT-based answering and summarization

FAISS — fast similarity search over PDF content

PyPDF2 — PDF text extraction
