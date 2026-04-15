import os
import time
import numpy as np
import faiss
import threading

from flask import Flask, render_template, request, redirect, url_for, send_file
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

load_dotenv()

app = Flask(__name__)

# ========================
# CONFIG
# ========================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ========================
# GLOBAL VARIABLES
# ========================
model = SentenceTransformer("all-MiniLM-L6-v2")

stored_chunks = None
stored_index = None
uploaded_filename = None
upload_time = None

chat_history = []
voice_history = []
last_summary = None

gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ========================
# AUTO DELETE FILES
# ========================
def check_expiry():
    global stored_chunks, stored_index, upload_time, uploaded_filename, last_summary
    global chat_history, voice_history

    current_time = time.time()

    # 🧠 Clear memory
    if upload_time and current_time - upload_time > 600:
        stored_chunks = None
        stored_index = None
        uploaded_filename = None
        last_summary = None
        chat_history = []
        voice_history = []

    # 🗂️ Delete old files
    if os.path.exists(UPLOAD_FOLDER):
        for file in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file)

            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)

                if file_age > 600:
                    try:
                        os.remove(file_path)
                    except:
                        pass

def auto_cleanup():
    while True:
        check_expiry()
        time.sleep(60)  # runs every 1 minute


# ========================
# FILE PROCESSING
# ========================
def extract_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, size=500):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def create_index(chunks):
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def search(query, index, chunks):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k=3)
    return " ".join([chunks[i] for i in I[0]])

# ========================
# GEMINI
# ========================
def generate_answer(question, context):
    prompt = f"Answer briefly:\nContext: {context}\nQuestion: {question}"

    try:
        # 🔹 First attempt
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response and response.text:
            cleaned = response.text.replace("\n", " ").strip()
            return " ".join(cleaned.split())

    except Exception as e:
        print("First attempt failed:", e)

        # 🔁 Retry once after 2 seconds
        time.sleep(2)

        try:
            print("Retrying...")
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            if response and response.text:
                cleaned = response.text.replace("\n", " ").strip()
                return " ".join(cleaned.split())

        except Exception as e2:
            print("Retry failed:", e2)

    # ❌ Final fallback
    return "⚠️ AI is busy right now. Please try again in a few seconds."

def generate_summary(mode):
    global stored_chunks

    # ✅ SAFETY CHECK (ADD THIS)
    if not stored_chunks:
        return "⚠️ No document available"
    
    prompts = {
        "short": "Summarize briefly.",
        "medium": "Summarize in moderate detail.",
        "detailed": "Summarize in detail."
    }

    try:
        # 🔥 IMPORTANT: pass document content
        text = " ".join(stored_chunks[:15])

        prompt = f"{prompts.get(mode)}\n\nDocument:\n{text}"

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if not response or not response.text:
            return "⚠️ Unable to generate summary. Try again."

        cleaned = response.text.strip()
        return cleaned

    except Exception as e:
        print("SUMMARY ERROR:", e)
        return "⚠️ Unable to generate summary. Please try again."

# ========================
# PDF GENERATOR
# ========================
def create_pdf(content, filename):
    path = os.path.join(UPLOAD_FOLDER, filename)

    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()

    flow = []
    for line in content:
        flow.append(Paragraph(line, styles["Normal"]))
        flow.append(Spacer(1, 10))

    doc.build(flow)
    return path

# ========================
# ROUTES
# ========================

@app.route("/", methods=["GET", "POST"])
def upload():
    global stored_chunks, stored_index, uploaded_filename, upload_time
    global chat_history, voice_history

    check_expiry()

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            text = extract_text(file_path)
            chunks = chunk_text(text)
            index = create_index(chunks)

            stored_chunks = chunks
            stored_index = index
            uploaded_filename = file.filename
            upload_time = time.time()

            # RESET histories
            chat_history = []
            voice_history = []

            return render_template("upload.html", success=True)

    return render_template("upload.html", success=False)

# ========================
# CHAT
# ========================
@app.route("/chat", methods=["GET", "POST"])
def chat():
    global chat_history

    check_expiry()

    if stored_index is None:
        return redirect(url_for("upload"))

    if request.method == "POST":
        q = request.form.get("query")

        if q:
            res = generate_answer(q, search(q, stored_index, stored_chunks))

            chat_history.append({"role": "user", "text": q})
            chat_history.append({"role": "ai", "text": res})

    is_new = request.method == "POST"

    return render_template(
        "chat.html",
        chat=chat_history,
        filename=uploaded_filename,
        is_new=is_new
    )

# ========================
# VOICE
# ========================
@app.route("/voice", methods=["GET", "POST"])
def voice():
    global voice_history

    check_expiry()

    if stored_chunks is None:
        return redirect(url_for("upload"))

    if request.method == "POST":
        q = request.form.get("query")

        if q:
            res = generate_answer(q, search(q, stored_index, stored_chunks))

            voice_history.append({"role": "user", "text": q})
            voice_history.append({"role": "ai", "text": res})

    return render_template("voice.html", chat=voice_history)

# ========================
# SUMMARY
# ========================
@app.route("/summary", methods=["GET", "POST"])
def summary():
    check_expiry()

    if stored_chunks is None:
        return redirect(url_for("upload"))

    result = None
    global last_summary

    if request.method == "POST":
        mode = request.form.get("mode")
        result = generate_summary(mode)
        last_summary = result

    return render_template("summary.html", result=result)

# ========================
# DOWNLOAD
# ========================
@app.route("/download_chat")
def download_chat():
    content = []
    for m in chat_history:
        role = "You" if m["role"] == "user" else "AI"
        content.append(f"<b>{role}:</b> {m['text']}")

    return send_file(create_pdf(content, "chat.pdf"), as_attachment=True)

@app.route("/download_summary")
def download_summary():
    global last_summary

    if not last_summary:
        return redirect(url_for("summary"))

    content = [f"<b>Summary:</b> {last_summary}"]

    return send_file(create_pdf(content, "summary.pdf"), as_attachment=True)

@app.route("/download_voice")
def download_voice():
    content = []
    for m in voice_history:
        role = "You" if m["role"] == "user" else "AI"
        content.append(f"<b>{role}:</b> {m['text']}")

    return send_file(create_pdf(content, "voice.pdf"), as_attachment=True)

# ========================
if __name__ == "__main__":
    # Start background cleaner
    threading.Thread(target=auto_cleanup, daemon=True).start()

    app.run(debug=False)