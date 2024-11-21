# Import necessary libraries
import pdfplumber
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import streamlit as st

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the generative AI model
rephrase_pipeline = pipeline("text2text-generation", model="t5-small")
chatbot_pipeline = pipeline("text2text-generation", model="t5-small")  # Can replace with more advanced chat model

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

# Function to calculate match percentage
def calculate_match_percentage(resume_skills, job_skills):
    resume_text = " ".join(resume_skills)
    job_text = " ".join(job_skills)
    vectorizer = CountVectorizer().fit_transform([resume_text, job_text])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return round(cosine_sim[0][1] * 100, 2)

# Function to rephrase text using generative AI
def rephrase_text(text):
    rephrased = rephrase_pipeline(text, max_length=100, num_return_sequences=1)
    return rephrased[0]['generated_text']

# Function to suggest additional skills
def suggest_skills(resume_skills, job_skills):
    missing_skills = set(job_skills) - set(resume_skills)
    if not missing_skills:
        return "No additional skills are needed! Your resume matches the job requirements well."
    return f"Consider adding these skills to your resume: {', '.join(missing_skills)}"

# Function to interact with a chatbot
def chatbot_response(query, resume_text):
    prompt = f"Resume Analysis Task: {query}\nResume Content: {resume_text}"
    response = chatbot_pipeline(prompt, max_length=150)
    return response[0]['generated_text']

# Function to provide improvement suggestions
def generate_improvement_suggestions(resume_skills, job_skills):
    suggestions = []
    if len(resume_skills) < len(job_skills):
        suggestions.append("Expand the skills section in your resume to cover the job requirements.")
    if len(set(resume_skills) & set(job_skills)) < len(job_skills) * 0.7:
        suggestions.append("Your resume does not match well with the job description. Add more relevant skills.")
    return suggestions if suggestions else ["Your resume is well-aligned with the job description!"]

# Streamlit app
st.title("Resume and Job Description Matcher")
st.write("Upload your resume PDF and input the job description to calculate the match percentage and get suggestions.")

# File upload
uploaded_file = st.file_uploader("Upload Resume PDF", type="pdf")
job_description = st.text_area("Enter Job Description")

if uploaded_file and job_description:
    # Process resume PDF
    with st.spinner("Extracting text from resume..."):
        resume_text = extract_text_from_pdf(uploaded_file)

    # Extract skills from resume and job description
    with st.spinner("Extracting skills..."):
        resume_skills = extract_skills(resume_text)
        job_skills = parse_job_description(job_description)

    # Calculate match percentage
    with st.spinner("Calculating match percentage..."):
        match_percentage = calculate_match_percentage(resume_skills, job_skills)

    # Skill suggestions
    with st.spinner("Generating skill suggestions..."):
        additional_skills = suggest_skills(resume_skills, job_skills)

    # Improvement suggestions
    with st.spinner("Analyzing improvements..."):
        improvement_suggestions = generate_improvement_suggestions(resume_skills, job_skills)

    # Display results
    st.subheader("Results")
    st.write(f"**Match Percentage:** {match_percentage}%")
    st.write("**Extracted Resume Skills:**", resume_skills)
    st.write("**Extracted Job Skills:**", job_skills)

    st.subheader("Suggestions")
    st.write("**Additional Skills to Add:**")
    st.write(additional_skills)
    st.write("**Improvement Suggestions:**")
    for suggestion in improvement_suggestions:
        st.write(f"- {suggestion}")

    # Chatbot
    st.subheader("Ask Your Resume")
    query = st.text_input("Enter a question about your resume (e.g., 'How can I improve my resume?')")
    if query:
        with st.spinner("Thinking..."):
            chatbot_reply = chatbot_response(query, resume_text)
        st.write("**Chatbot Response:**")
        st.write(chatbot_reply)
else:
    st.warning("Please upload a resume and provide a job description.")