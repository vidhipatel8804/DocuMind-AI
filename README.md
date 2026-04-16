# 📄 DocuMind AI – Intelligent Document Assistant

DocuMind AI is an AI-powered document interaction system that allows users to upload documents and extract insights through chat, summarization, and voice-based interaction.

The system uses a **context-based retrieval approach** combined with **Google Gemini API** to generate accurate, context-aware responses from document content.

---

## 🚀 Features

- 📂 Upload and process PDF documents  
- 💬 Document-based Question Answering (Chat)  
- 🧠 Context-aware AI responses using Gemini  
- 📝 Document Summarization  
  - Concise  
  - Standard  
  - Detailed  
- 🎙️ Voice-based Query Interaction  
- 📥 Download responses as PDF (Chat / Summary / Voice)  
- ⚡ Fast and lightweight processing  

---

## 🧠 Working Pipeline

1. Upload document (PDF)
2. Extract text using PyPDF
3. Split text into smaller chunks
4. Store processed chunks
5. User asks a question
6. Relevant content identified using text-based matching
7. Context + query sent to Gemini API
8. AI generates response
9. Response displayed to user

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Backend:** Flask  
- **Frontend:** HTML, CSS, JavaScript  
- **AI Model:** Google Gemini API  
- **Core Concept:** Context-Based Retrieval  
- **Libraries:** PyPDF, ReportLab  

---

## 🌐 Live Demo

👉 https://documind-ai-bgq0.onrender.com 

---
