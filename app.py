from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from PyPDF2 import PdfReader
import docx
import sqlite3
import os
from langdetect import detect, DetectorFactory
from ddgs import DDGS
import hashlib
import torch

import re

DetectorFactory.seed = 0  # reproducible language detection

app = Flask(__name__)

# ===================== Cache for scraped results =====================
scrape_cache = {}  # key: text hash, value: scraped results

# ===================== Load Models =====================
try:
    plagiarism_model = SentenceTransformer('paraphrase-mpnet-base-v2')
    ai_detector = pipeline("text-classification", model="roberta-large-openai-detector")
    rephrase_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    rephrase_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    rephrase_model.to('cpu')  # change to 'cuda' if GPU available
except Exception as e:
    print(f"Model loading error: {e}")
    plagiarism_model = None
    ai_detector = None
    rephrase_model = None

# ===================== Reference Corpus =====================
# NOTE: The static reference corpus was removed because vague generic
# sentences ("sample text", "copied content") caused massive false
# positives — any text about common topics scored 40-70% plagiarism.
# Plagiarism is now scored exclusively against live web-scraped sources,
# which represent actual published content the user might have copied from.

# ===================== SQLite Database =====================
if not os.path.exists('plagiarism.db'):
    conn = sqlite3.connect('plagiarism.db')
    conn.execute('''CREATE TABLE reports
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     text TEXT,
                     plagiarism_score REAL,
                     ai_confidence REAL,
                     language TEXT,
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

# ===================== Helper Functions =====================
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def extract_key_sentences(text, max_sentences=3):
    """Extract the most representative sentences for web search queries.
    DuckDuckGo works best with short, focused queries (5-15 words),
    not 400-character blobs of raw text."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter to meaningful sentences (at least 5 words, skip very short ones)
    meaningful = [s.strip() for s in sentences if len(s.split()) >= 5]
    return meaningful[:max_sentences] if meaningful else [text[:150]]

def scrape_web_for_similar(text, max_results=5):
    """Search the web using focused key-sentence queries.
    Returns two parallel lists:
      - plain_snippets:  clean text for embedding comparison
      - display_sources: HTML-formatted strings for the UI
    """
    key_sentences = extract_key_sentences(text, max_sentences=3)
    cache_key = hashlib.md5("|".join(key_sentences).encode()).hexdigest()
    if cache_key in scrape_cache:
        return scrape_cache[cache_key]

    try:
        plain_snippets = []
        display_sources = []
        seen_snippets = set()
        with DDGS() as ddgs:
            for query in key_sentences:
                search_results = ddgs.text(query, max_results=max_results)
                for r in search_results:
                    snippet = r.get("body", "")
                    link = r.get("href", "")
                    if snippet and len(snippet) > 50 and snippet not in seen_snippets:
                        try:
                            if detect(snippet) == 'en':
                                seen_snippets.add(snippet)
                                plain_snippets.append(snippet)
                                display_sources.append(
                                    f"{snippet} - <a href='{link}' target='_blank'>source</a>"
                                )
                        except:
                            continue
        if not plain_snippets:
            plain_snippets = []
            display_sources = ["No relevant content found."]
        result = (plain_snippets, display_sources)
        scrape_cache[cache_key] = result
        return result
    except Exception as e:
        print(f"Web scraping error: {e}")
        return ([], ["Error fetching web content."])

def rephrase_content(text):
    input_text = "paraphrase: " + text + " </s>"
    inputs = rephrase_tokenizer(
        [input_text],
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    )
    outputs = rephrase_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=256,
        min_length=50,
        num_beams=5,
        num_return_sequences=1,
        early_stopping=True,
        no_repeat_ngram_size=3,
        temperature=0.7,
        top_p=0.9
    )
    return rephrase_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# ===================== Flask Routes =====================
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/check', methods=['POST'])
def do_check():
    text = request.form.get('text', '').strip()
    file = request.files.get('file')

    if not text and not file:
        return jsonify({'error': 'Please provide text or upload a file!'}), 400

    # ----------- Extract text from file -----------
    if file:
        try:
            if file.filename.endswith('.pdf'):
                pdf_reader = PdfReader(file)
                text = "".join([page.extract_text() or '' for page in pdf_reader.pages])
            elif file.filename.endswith('.docx'):
                doc = docx.Document(file)
                text = "\n".join([para.text for para in doc.paragraphs])
            elif file.filename.endswith('.txt'):
                text = file.read().decode('utf-8')
            else:
                return jsonify({'error': 'Only PDF, DOCX, or TXT files allowed!'}), 400
        except Exception as e:
            return jsonify({'error': f'Error reading file: {e}'}), 400

    if not text.strip():
        return jsonify({'error': 'The provided text is empty!'}), 400

    # ----------- Language detection -----------
    try:
        language = detect(text)
    except Exception as e:
        print(f"Language detection error: {e}")
        language = "unknown"

    # ----------- Scrape web for similar content -----------
    plain_snippets, display_sources = scrape_web_for_similar(text)

    # ----------- Chunk text for long documents -----------
    text_chunks = chunk_text(text)

    # ----------- Plagiarism check -----------
    if plagiarism_model is None:
        return jsonify({'error': 'Plagiarism model not loaded'}), 500

    try:
        if plain_snippets:
            user_embeddings = [plagiarism_model.encode(chunk, convert_to_tensor=True) for chunk in text_chunks]
            # Encode only clean text (no HTML) for accurate similarity
            reference_embeddings = plagiarism_model.encode(plain_snippets, convert_to_tensor=True)
            cosine_scores = [util.pytorch_cos_sim(ue, reference_embeddings)[0] for ue in user_embeddings]

            # Weighted average: each chunk's best match, weighted by chunk word-count.
            # This prevents a single similar sentence from inflating the entire score.
            chunk_best_scores = [cs.max().item() for cs in cosine_scores]
            chunk_weights = [len(chunk.split()) for chunk in text_chunks]
            total_weight = sum(chunk_weights)
            plagiarism_score = (
                sum(s * w for s, w in zip(chunk_best_scores, chunk_weights)) / total_weight
            ) * 100
        else:
            # No web sources found — cannot determine plagiarism from web
            plagiarism_score = 0.0
    except Exception as e:
        print(f"Plagiarism check error: {e}")
        return jsonify({'error': 'Failed to calculate plagiarism score'}), 500

    # ----------- AI content detection (all chunks) -----------
    if ai_detector is None:
        return jsonify({'error': 'AI detector not loaded'}), 500

    try:
        # roberta-large-openai-detector labels:
        #   LABEL_0 = Real (human-written)
        #   LABEL_1 = Fake (AI-generated)
        # Analyze ALL chunks so long documents aren't judged by the intro alone.
        chunk_ai_probs = []
        for chunk in text_chunks:
            truncated = chunk[:512]  # model max token window
            result = ai_detector(truncated)[0]
            if result['label'] == 'LABEL_1':  # Fake / AI
                chunk_ai_probs.append(result['score'])
            else:  # LABEL_0 = Real → invert to get P(AI)
                chunk_ai_probs.append(1.0 - result['score'])

        # Weighted average across chunks (by word count)
        chunk_weights = [len(chunk.split()) for chunk in text_chunks]
        total_weight = sum(chunk_weights)
        avg_ai_prob = sum(p * w for p, w in zip(chunk_ai_probs, chunk_weights)) / total_weight

        ai_confidence = round(avg_ai_prob * 100, 2)
        is_ai_generated = avg_ai_prob > 0.7
    except Exception as e:
        print(f"AI detection error: {e}")
        is_ai_generated = False
        ai_confidence = 0.0

    # ----------- Rephrase if plagiarism >70% -----------
    rephrased = None
    if plagiarism_score > 70 and rephrase_model is not None:
        rephrased = rephrase_content(text)

    # ----------- Save report to database -----------
    try:
        conn = sqlite3.connect('plagiarism.db')
        conn.execute(
            "INSERT INTO reports (text, plagiarism_score, ai_confidence, language) VALUES (?, ?, ?, ?)",
            (text[:500], plagiarism_score, ai_confidence, language)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")

    # ----------- Messages -----------
    message = 'Possible plagiarism detected!' if plagiarism_score > 70 else 'No plagiarism detected!'
    ai_message = 'AI-generated content detected!' if is_ai_generated else 'No AI-generated content detected!'

    return jsonify({
        'message': f"{message} (Score: {round(plagiarism_score, 2)})",
        'ai_message': ai_message,
        'ai_confidence': round(ai_confidence, 2),
        'language': language,
        'rephrased': rephrased,
        'scraped_sources': display_sources
    })


# ===================== Reports Endpoint =====================
@app.route('/reports')
def get_reports():
    try:
        conn = sqlite3.connect('plagiarism.db')
        cursor = conn.execute("SELECT * FROM reports ORDER BY timestamp DESC LIMIT 10")
        reports = [
            {'id': row[0], 'text': row[1], 'plagiarism_score': row[2],
             'ai_confidence': row[3], 'language': row[4], 'timestamp': row[5]}
            for row in cursor.fetchall()
        ]
        conn.close()
        return jsonify(reports)
    except Exception as e:
        print(f"Error in /reports: {e}")
        return jsonify({'error': 'Failed to load reports'}), 500


# ===================== Contact Form =====================
@app.route('/contact', methods=['POST'])
def contact():
    name = request.form.get('name')
    email = request.form.get('email')
    message = request.form.get('message')

    if not name or not email or not message:
        return jsonify({'error': 'Name, email, and message required!'}), 400

    return jsonify({'message': f'Thank you, {name}! Your message has been received. We will contact you at {email}.'})


# ===================== Run Flask =====================
if __name__ == '__main__':
    app.run(debug=True)
