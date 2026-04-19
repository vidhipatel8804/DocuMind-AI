import os
import re
import time
import threading
import uuid

from flask import Flask, render_template, request, redirect, url_for, send_file, session
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24))

# ========================
# CONFIG
# ========================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ========================
# SESSIONS & VARIABLES
# ========================
user_sessions = {}

def get_session_data():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    
    sid = session["session_id"]
    if sid not in user_sessions:
        user_sessions[sid] = {
            "stored_chunks": None,
            "uploaded_filename": None,
            "upload_time": None,
            "chat_history": [],
            "voice_history": [],
            "last_summary": None
        }
    return user_sessions[sid]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "with", "you",
    "your"
}

gemini_client = None

# ========================
# AUTO DELETE FILES
# ========================
def check_expiry():
    current_time = time.time()

    # 🧠 Clear memory
    expired_sessions = []
    for sid, data in list(user_sessions.items()):
        if data.get("upload_time") and current_time - data["upload_time"] > 600:
            expired_sessions.append(sid)
            
    for sid in expired_sessions:
        user_sessions.pop(sid, None)

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

def tokenize(text):
    return [
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    ]

def build_context(chunks, max_chars=16000):
    context_parts = []
    total_chars = 0

    for chunk in chunks:
        remaining = max_chars - total_chars
        if remaining <= 0:
            break

        snippet = chunk[:remaining]
        context_parts.append(snippet)
        total_chars += len(snippet)

    return "\n\n".join(context_parts)

def get_gemini_client():
    global gemini_client

    if gemini_client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GOOGLE_API_KEY. Add it to your environment before using chat or summary features."
            )

        gemini_client = genai.Client(api_key=api_key)

    return gemini_client

def search(query, chunks, top_k=3, max_chars=12000):
    if not chunks:
        return ""

    query_terms = tokenize(query)
    if not query_terms:
        return build_context(chunks[:top_k], max_chars=max_chars)

    scored_chunks = []
    unique_terms = set(query_terms)

    for index, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        unique_hits = sum(1 for term in unique_terms if term in chunk_lower)
        frequency_hits = sum(chunk_lower.count(term) for term in query_terms)
        score = unique_hits * 10 + frequency_hits

        if score > 0:
            scored_chunks.append((score, index, chunk))

    if not scored_chunks:
        selected_chunks = chunks[:top_k]
    else:
        scored_chunks.sort(key=lambda item: (-item[0], item[1]))
        selected_chunks = [chunk for _, _, chunk in scored_chunks[:top_k]]

    return build_context(selected_chunks, max_chars=max_chars)

# ========================
# GEMINI
# ========================
def generate_answer(question, context):
    prompt = (
        "Answer the question using only the document context below. "
        "If the answer is not in the document, say that clearly in one short sentence.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    try:
        # 🔹 First attempt
        response = get_gemini_client().models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response and response.text:
            cleaned = response.text.replace("\n", " ").strip()
            return " ".join(cleaned.split())

    except RuntimeError as exc:
        return str(exc)
    except Exception as e:
        print("First attempt failed:", e)

        # 🔁 Retry once after 2 seconds
        time.sleep(2)

        try:
            print("Retrying...")
            response = get_gemini_client().models.generate_content(
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

def generate_summary(mode, chunks):
    # ✅ SAFETY CHECK (ADD THIS)
    if not chunks:
        return "⚠️ No document available"
    
    prompts = {
        "short": "Summarize briefly.",
        "medium": "Summarize in moderate detail.",
        "detailed": "Summarize in detail."
    }

    try:
        # 🔥 IMPORTANT: pass document content
        text = build_context(chunks, max_chars=18000)

        prompt = f"{prompts.get(mode)}\n\nDocument:\n{text}"

        response = get_gemini_client().models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if not response or not response.text:
            return "⚠️ Unable to generate summary. Try again."

        cleaned = response.text.strip()
        return cleaned

    except RuntimeError as exc:
        return str(exc)
    except Exception as e:
        print("SUMMARY ERROR:", e)
        return "⚠️ Unable to generate summary. Please try again."

# ========================
# PDF GENERATOR
# ========================
def create_pdf(content, filename):
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PDF export requires the 'reportlab' package. Install it with "
            "'pip install -r requirements.txt' or 'pip install reportlab'."
        ) from exc

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
    session_data = get_session_data()
    check_expiry()

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            text = extract_text(file_path)
            chunks = chunk_text(text)

            session_data["stored_chunks"] = chunks
            session_data["uploaded_filename"] = file.filename
            session_data["upload_time"] = time.time()

            # RESET histories
            session_data["chat_history"] = []
            session_data["voice_history"] = []

            return render_template("upload.html", success=True)

    return render_template("upload.html", success=False)

# ========================
# CHAT
# ========================
@app.route("/chat", methods=["GET", "POST"])
def chat():
    session_data = get_session_data()
    check_expiry()

    if session_data["stored_chunks"] is None:
        return redirect(url_for("upload"))

    if request.method == "POST":
        q = request.form.get("query")

        if q:
            res = generate_answer(q, search(q, session_data["stored_chunks"]))

            session_data["chat_history"].append({"role": "user", "text": q})
            session_data["chat_history"].append({"role": "ai", "text": res})

    is_new = request.method == "POST"

    return render_template(
        "chat.html",
        chat=session_data["chat_history"],
        filename=session_data["uploaded_filename"],
        is_new=is_new
    )

# ========================
# VOICE
# ========================
@app.route("/voice", methods=["GET", "POST"])
def voice():
    session_data = get_session_data()
    check_expiry()

    if session_data["stored_chunks"] is None:
        return redirect(url_for("upload"))

    if request.method == "POST":
        q = request.form.get("query")

        if q:
            res = generate_answer(q, search(q, session_data["stored_chunks"]))

            session_data["voice_history"].append({"role": "user", "text": q})
            session_data["voice_history"].append({"role": "ai", "text": res})

    return render_template("voice.html", chat=session_data["voice_history"])

# ========================
# SUMMARY
# ========================
@app.route("/summary", methods=["GET", "POST"])
def summary():
    session_data = get_session_data()
    check_expiry()

    if session_data["stored_chunks"] is None:
        return redirect(url_for("upload"))

    result = None

    if request.method == "POST":
        mode = request.form.get("mode")
        result = generate_summary(mode, session_data["stored_chunks"])
        session_data["last_summary"] = result

    return render_template("summary.html", result=result)

# ========================
# DOWNLOAD
# ========================
@app.route("/download_chat")
def download_chat():
    session_data = get_session_data()
    content = []
    for m in session_data["chat_history"]:
        role = "You" if m["role"] == "user" else "AI"
        content.append(f"<b>{role}:</b> {m['text']}")

    try:
        return send_file(create_pdf(content, "chat.pdf"), as_attachment=True)
    except RuntimeError as exc:
        return str(exc), 503

@app.route("/download_summary")
def download_summary():
    session_data = get_session_data()

    if not session_data["last_summary"]:
        return redirect(url_for("summary"))

    content = [f"<b>Summary:</b> {session_data['last_summary']}"]

    try:
        return send_file(create_pdf(content, "summary.pdf"), as_attachment=True)
    except RuntimeError as exc:
        return str(exc), 503

@app.route("/download_voice")
def download_voice():
    session_data = get_session_data()
    content = []
    for m in session_data["voice_history"]:
        role = "You" if m["role"] == "user" else "AI"
        content.append(f"<b>{role}:</b> {m['text']}")

    try:
        return send_file(create_pdf(content, "voice.pdf"), as_attachment=True)
    except RuntimeError as exc:
        return str(exc), 503

# ========================
# START BACKGROUND CLEANER (always runs)
threading.Thread(target=auto_cleanup, daemon=True).start()

# ========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
