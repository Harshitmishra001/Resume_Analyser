import os
import json
import re
import fitz  # PyMuPDF for better PDF text extraction
from flask import Flask, request, jsonify, render_template, redirect
import google.generativeai as genai
from dotenv import load_dotenv
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Load environment variables
load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-pro")

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

# Initialize Sentence Transformer and spaCy models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load('en_core_web_sm')

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load skill dataset from JSON
def load_skills(file_path="skills.json"):
    """Load the skill dataset from a JSON file."""
    with open(file_path, "r") as file:
        data = json.load(file)
    return set(skill.lower() for skill in data.get("skills", []))

SKILLS = load_skills()

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file using fitz (PyMuPDF)."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text.strip()

def parse_job_description(job_description):
    """Extract required skills and other details from the job description."""
    doc = nlp(job_description)
    required_skills = []

    # Match skills from the loaded dataset
    for token in doc:
        if token.text.lower() in SKILLS:
            required_skills.append(token.text.lower())

    qualifications = []
    responsibilities = []

    for sent in doc.sents:
        if any(keyword in sent.text.lower() for keyword in ['qualification', 'degree', 'education']):
            qualifications.append(sent.text.strip())
        if any(keyword in sent.text.lower() for keyword in ['responsibility', 'responsible', 'task']):
            responsibilities.append(sent.text.strip())

    return {
        'required_skills': ', '.join(set(required_skills)),
        'qualifications': ' '.join(qualifications),
        'responsibilities': ' '.join(responsibilities)
    }

def extract_details_with_ai(text):
    """Extract key details from resume text using Generative AI."""
    prompt = f"""
    Extract details from the following resume text. Respond in JSON format with these fields:
    - "name"
    - "highest_qualification"
    - "skills"
    - "total_experience_years"
    - "projects"
    If a field is missing, return "NA".

    Resume Text: {text}
    """
    try:
        response = model.generate_content([prompt])
        if response and response.text.strip():
            return json.loads(response.text.strip())
        else:
            raise ValueError("Empty response from Generative AI")
    except Exception as e:
        print("Error extracting details with AI:", e)
        return {
            "error": "The AI service is currently busy or unavailable. Please try again later."
        }

def compute_similarity(resume_text, jd_text):
    """Compute semantic similarity between resume and job description."""
    resume_combined = ' '.join(resume_text.values())
    jd_combined = ' '.join(jd_text.values())

    resume_embedding = sentence_model.encode(resume_combined)
    jd_embedding = sentence_model.encode(jd_combined)

    similarity = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
    return similarity * 100  # Convert to percentage

def identify_gaps(resume_text, jd_text):
    """Identify gaps between resume and job description."""
    resume_skills = set(resume_text.get('skills', '').lower().split(', '))
    jd_skills = set(jd_text.get('required_skills', '').lower().split(', '))

    missing_skills = jd_skills - resume_skills
    return list(missing_skills)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle resume upload
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and file.filename.lower().endswith('.pdf'):
            filename = "data.pdf"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Extract text from resume
            resume_text_raw = extract_text_from_pdf(filepath)
            resume_details = extract_details_with_ai(resume_text_raw)

            # Handle job description input
            job_description = request.form.get('job_description', '')
            if not job_description:
                feedback = {
                    'match_score': "NA",
                    'gaps': ["No job description provided."]
                }
            else:
                parsed_jd = parse_job_description(job_description)
                match_score = compute_similarity(resume_details, parsed_jd)
                gaps = identify_gaps(resume_details, parsed_jd)
                feedback = {
                    'match_score': round(match_score, 2),
                    'gaps': gaps if gaps else ["No gaps identified. Your resume matches the job description well!"]
                }

            # Clean up uploaded file
            os.remove(filepath)

            return render_template('index.html', details=resume_details, feedback=feedback)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)