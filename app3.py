# Import necessary libraries
import pdfplumber
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import streamlit as st
from transformers import pipeline

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load generative AI model
rephrase_pipeline = pipeline("text2text-generation", model="t5-small")

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
    st.pyplot(plt)

# Streamlit app
st.title("Enhanced Resume and Job Description Matcher")
st.write("Upload your resume PDF and input the job description to calculate the match percentage, get suggestions, and visualize results.")

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

    # Display results
    st.subheader("Results")
    st.write(f"**Match Percentage:** {match_percentage}%")
    st.write("**Extracted Resume Skills:**", resume_skills)
    st.write("**Extracted Job Skills:**", job_skills)

    st.subheader("Suggestions")
    st.write("**Additional Skills to Add:**")
    st.write(additional_skills)

    # Visualization
    st.subheader("Skill Match Visualization")
    plot_skill_match(resume_skills, job_skills)
else:
    st.warning("Please upload a resume and provide a job description.")