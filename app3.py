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

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load DialoGPT model and tokenizer for chatbot
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
chatbot_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

app = Flask(__name__)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to extract skills from text
def extract_skills(text):
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["SKILL", "PERSON", "ORG"]]
    return skills

# Function to parse job description and extract skills
def parse_job_description(job_description):
    doc = nlp(job_description)
    job_skills = [ent.text for ent in doc.ents if ent.label_ in ["SKILL", "PERSON", "ORG"]]
    return job_skills

# Function to calculate match percentage using TF-IDF
def calculate_match_percentage(resume_skills, job_skills):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([" ".join(resume_skills), " ".join(job_skills)])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]).flatten()
    return round(cosine_sim[0] * 100, 2)

# Function to suggest additional skills
def suggest_skills(resume_skills, job_skills):
    missing_skills = set(job_skills) - set(resume_skills)
    if not missing_skills:
        return "No additional skills are needed! Your resume matches the job requirements well."
    return f"Consider adding these skills to your resume: {', '.join(missing_skills)}"

# Function to suggest remove skills
def suggest_reskills(resume_skills, job_skills):
    missing_skills = set(resume_skills) - set(job_skills)
    if not missing_skills:
        return "No additional skills are needed! Your resume matches the job requirements well."
    return f"Consider removing these skills from your resume: {', '.join(missing_skills)}"

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

# Function to generate chatbot response using DialoGPT
def chatbot_response(user_input, chat_history_ids=None):
    # Encode the user input and append to chat history
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1)
    else:
        bot_input_ids = input_ids
    
    # Generate response
    chat_history_ids = chatbot_model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return bot_response, chat_history_ids

# Route to home page
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['resumeFile']
        job_description = request.form['job_description']
        
        if uploaded_file and job_description:
            # Process resume PDF
            resume_text = extract_text_from_pdf(uploaded_file)

            # Extract skills from resume and job description
            resume_skills = extract_skills(resume_text)
            job_skills = parse_job_description(job_description)

            # Calculate match percentage
            match_percentage = calculate_match_percentage(resume_skills, job_skills)

            # Skill suggestions
            additional_skills = suggest_skills(resume_skills, job_skills)
            additional_remskills = suggest_reskills(resume_skills, job_skills)

            # Visualization
            plot_image = plot_skill_match(resume_skills, job_skills)

            # Return the result to the template
            return render_template('results.html', match_percentage=match_percentage, resume_skills=resume_skills,
                                job_skills=job_skills, additional_skills=additional_skills, additional_remskills=additional_remskills,
                                plot_image=plot_image)

    return render_template('index.html')

# Route to handle chatbot messages
@app.route('/chatbot', methods=['POST'])
def chatbot():
    chat_input = request.json.get('message')
    if chat_input:
        chatbot_reply, chat_history_ids = chatbot_response(chat_input, None)
        return jsonify({'response': chatbot_reply})

if __name__ == '__main__':
    app.run(debug=True)
