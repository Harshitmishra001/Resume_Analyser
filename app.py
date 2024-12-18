from flask import Flask, render_template, request, jsonify
import pdfplumber
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import io
from chat import get_response
import json
# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load DialoGPT model and tokenizer for chatbot
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
chatbot_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Dictionary of synonyms for skills
skill_synonyms = {
    "project management": ["project manager", "managing projects", "project lead"],
    "data analysis": ["data analyst", "analyzing data", "data interpretation"],
    # Add more synonyms as needed
}

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to normalize skills
def normalize_skills(skills):
    normalized = set()
    for skill in skills:
        skill_lower = skill.lower()  # Normalize to lowercase
        found = False
        for key, synonyms in skill_synonyms.items():
            if skill_lower == key.lower() or skill_lower in map(str.lower, synonyms):
                normalized.add(key.lower())  # Store normalized key in lowercase
                found = True
                break
        if not found:
            normalized.add(skill_lower)  # Add the skill in lowercase
    return normalized

# Function to extract skills from text
def extract_skills(text):
    doc = nlp(text)
    skills = [ent.text.lower() for ent in doc.ents if ent.label_ in ["SKILL", "PERSON", "ORG", "CERTIFICATION", "EDUCATION"]]
    return skills

# Function to parse job description and extract skills
def parse_job_description(job_description):
    doc = nlp(job_description)
    job_skills = [ent.text for ent in doc.ents if ent.label_ in ["SKILL", "PERSON", "ORG", "CERTIFICATION", "EDUCATION"]]
    return job_skills

# Function to calculate weighted match percentage

def calculate_weighted_match_percentage(resume_skills, job_skills):
    # Convert resume_skills to a list for counting
    resume_skills_list = list(resume_skills)
    
    # Convert job_skills to a list to count occurrences
    job_skills_list = list(job_skills)
    
    # Count occurrences of each skill in job_skills
    job_skill_counts = {skill: job_skills_list.count(skill) for skill in set(job_skills_list)}
    
    # Calculate weighted match
    match_score = sum(job_skill_counts[skill] for skill in resume_skills_list if skill in job_skill_counts)
    total_weight = sum(job_skill_counts.values())
    
    return round((match_score / total_weight) * 100, 2) if total_weight > 0 else 0
# Function to suggest additional skills
def suggest_skills(resume_skills, job_skills):
    missing_skills = set(job_skills) - set(resume_skills)
    if not missing_skills:
        return "No additional skills are needed! Your resume matches the job requirements well."
    return f"Consider adding these skills to your resume: {', '.join(missing_skills)}"

def prepare_missing_skills(resume_skills, job_skills):
    # Load skill-resource mapping from skills.json
    with open("skills.json", "r") as f:
        skills = json.load(f)

    missing_skills = set(job_skills) - set(resume_skills)
    recommendations = []

    # Find resources for each missing skill
    for skill in missing_skills:
        for skill_entry in skills['skills']:
            if skill_entry['skill'] == skill:
                resource = skill_entry['resources'][0].get('link', 'No resources available')  # Get the first resource link
                recommendations.append(f"{skill} : {resource}")  # Format as 'skill : <link>'

    # Prepare data for HTML
    if not recommendations:
        return {"message": "Your resume matches the job requirements perfectly!"}
    
    return {"recommendations": recommendations}




# Function to visualize skill match
def plot_skill_match(resume_skills, job_skills):
    common_skills = set(resume_skills).intersection(set(job_skills))
    missing_skills = set(job_skills) - set(resume_skills)
    data = {
        'Matched Skills': len(common_skills),
        'Missing Skills': len(missing_skills)
    }
    plt.bar(data.keys(), data.values(), color=['green', 'red'])
    plt.title('Skill Match Overview')
    plt.xlabel('Skill Categories')
    plt.ylabel('Number of Skills')
    plt.tight_layout()
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    plt.close(fig)
    return buf

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['resumeFile']
        job_description = request.form['job_description']
        user_skills = request.form.get('user_skills', '').split(',')

        if uploaded_file and job_description:
            # Process resume PDF
            resume_text = extract_text_from_pdf(uploaded_file)

            # Extract and normalize skills
            resume_skills = normalize_skills(extract_skills(resume_text))
            job_skills = normalize_skills(parse_job_description(job_description))

            # Include user input skills
            resume_skills.update(normalize_skills(user_skills))

            # Calculate match percentage
            match_percentage = calculate_weighted_match_percentage(resume_skills, job_skills)

            # Skill suggestions and visualization
            additional_skills = prepare_missing_skills(resume_skills, job_skills)
            plot_image = plot_skill_match(resume_skills, job_skills)

            return render_template('results.html', match_percentage=match_percentage, resume_skills=resume_skills,
                                job_skills=job_skills, additional_skills=additional_skills["recommendations"], plot_image=plot_image)

    return render_template('index.html')

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == '__main__':
    app.run(debug=True)