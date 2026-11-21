
from flask import Flask, render_template, request, jsonify
import os
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types

# -------------------------------
# Flask Application
# -------------------------------
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Please check your .env file."
    )

# Initialize Gemini client
client = genai.Client(api_key=api_key)

# Model configuration
MODEL_NAME = "gemini-4-hydrogen"


# -------------------------------
# Reusable Gemini Function
# -------------------------------
def generate_response(prompt, system_instruction):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=1024
        )
    )

    if not response.text:
        raise ValueError("The model returned an empty response.")

    return response.text.strip()


# -------------------------------
# Home Route
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------
# AI Personal Assistant
# -------------------------------
@app.route("/ask", methods=["POST"])
def ask():

    question = request.form.get("question", "").strip()

    # Validate user input
    if not question:
        return jsonify({
            "error": "Please enter a question."
        }), 400

    try:
        answer = generate_response(
            prompt=question,
            system_instruction=(
                "You are a helpful, intelligent and friendly AI personal "
                "assistant. Answer questions clearly, accurately and in a "
                "user-friendly manner. Explain complex concepts simply. "
                "Use structured answers when helpful. If you are unsure "
                "about something, be honest rather than inventing facts."
            )
        )

        return jsonify({
            "response": answer
        }), 200

    except Exception:
        logger.exception("Error in AI assistant")

    return jsonify({
        "error": "Unable to generate a response. Please try again."
    }), 502


# -------------------------------
# Email Summarizer
# -------------------------------
@app.route("/summarize", methods=["POST"])
def summarize():

    email_text = request.form.get("email", "").strip()

    # Validate email content
    if not email_text:
        return jsonify({
            "error": "Please enter the email text."
        }), 400

    prompt = f"""
    Summarize the following email in 2-3 sentences.

    Focus on:
    - Main purpose of the email
    - Important information
    - Actions or deadlines, if any

    Email:
    {email_text}
    """

    try:
        summary = generate_response(
            prompt=prompt,
            system_instruction=(
                "You are an expert email summarization assistant. "
                "Create concise, accurate and professional summaries. "
                "Do not add information that is not present in the email."
            )
        )

        return jsonify({
            "response": summary
        }), 200

    except Exception as e:
        logger.exception("Error while summarizing email")

        return jsonify({
            "error": "Unable to summarize the email. Please try again."
        }), 502


# -------------------------------
# Error Handling
# -------------------------------
@app.errorhandler(404)
def page_not_found(error):
    return jsonify({
        "error": "Page not found."
    }), 404


# -------------------------------
# Run Application
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
