
# 🏥 Medical Chatbot using LangChain, Pinecone & LLaMA

A smart medical assistant chatbot built with:
- 🧠 LLaMA-2 7B quantized model (locally hosted)
- 🔍 LangChain for chaining queries and prompt templates
- 📚 Pinecone for vector search on medical PDFs
- 🌐 Flask backend with a simple chat UI

---

## 🚀 Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/Prokandera/Medical-chatbot-.git
cd Medical-chatbot-
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download or place your model**
Put `llama-2-7b-chat.ggmlv3.q4_0.bin` inside `model/`

4. **Set environment variables**
Create a `.env` file using `.env.example`

5. **Index your PDFs**
```bash
python storeindex.py
```

6. **Run the server**
```bash
python app.py
```

Go to `http://localhost:8080` to chat!

---

## 📁 Folder Structure
```
Medical-chatbot-/
│
├── model/                          # Place your .bin LLaMA model here
├── data/                           # Place your PDFs here
├── templates/
│   └── chat.html                   # Frontend
├── src/
│   ├── helper.py                   # PDF, text splitter, embeddings
│   └── prompt.py                   # Custom prompt template
│
├── storeindex.py                   # Loads and stores vectors in Pinecone
├── app.py                          # Main Flask app
├── .env                            # API keys (not tracked)
├── .env.example                    # Template for .env
├── requirements.txt                # Python packages
└── README.md                       # You’re reading it
```

---

## 📌 Environment Variables

Create a `.env` file with:

```dotenv
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_API_ENV=your_pinecone_environment
```

---

## 🛠 Technologies Used
- LangChain
- LLaMA 2 (ggml quantized)
- Pinecone
- Flask
- HuggingFace Transformers

---

## 🧠 Future Improvements
- Add user authentication
- Expand with more medical datasets
- Add voice input/output

---

Made with ❤️ by Abhishek Kandera
