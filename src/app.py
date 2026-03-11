from flask import Flask, request, jsonify, send_from_directory
import random
import os 
import fitz
from typing import cast

from src import prediction



app = Flask(__name__, static_folder="../frontend", static_url_path="/")

@app.route("/")
def webpage():
    if (app.static_folder is None):
        return jsonify({"error": "Internal server error"}), 500
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_files(path):
    if (app.static_folder is None):
        return jsonify({"error": "Internal server error"}), 500
    return send_from_directory(app.static_folder, path)

@app.route("/api/plaintext", methods=["POST"])
def plaintext():
    body = request.get_json(silent=True) or {}
    text = body.get("text", "")
    print("Got text length:", len(text))
    score_bs, score_ai = prediction.scoreWhole(text)
    return jsonify({"score_bs": score_bs, "score_ai": score_ai}), 200

@app.route("/api/analyse", methods=["POST"])
def analyse():
    body = request.get_json(silent=True) or {}
    text = body.get("text", "")
    overall_bs, overall_ai = prediction.scoreWhole(text)
    bs_highlight, ai_highlight = prediction.scoreSentences(text, overall_bs, overall_ai)
    print("hello \n \n \n \n \n hello")
    print(bs_highlight)
    print(ai_highlight)
    print(overall_bs)
    print(overall_ai)
    return jsonify({"overall_bs": overall_bs, "bs_highlight": bs_highlight, "overall_ai": overall_ai, "ai_highlight": ai_highlight}), 200


@app.route("/api/pdf", methods=["POST"])
def pdf():
    file = request.files["file"]
    if (not file or not file.filename):
        return jsonify({"error": "No file part"}), 400
    
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Invalid file type"}), 400
    
    pdf_bytes = file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += cast(str, page.get_text())

    return jsonify({"text": text}), 200

if (__name__ == "__main__"):
    app.run()
