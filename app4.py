# Import necessary libraries
import pdfplumber
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load DialoGPT model and tokenizer for chatbot
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
chatbot_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

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
    st.pyplot(plt)

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

# Streamlit app
st.title("Enhanced Resume and Job Description Matcher with Chatbot")
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
    with st.spinner("Generating skill suggestions..."):
        additional_remskills = suggest_reskills(resume_skills, job_skills)
    # Display results
    st.subheader("Results")
    st.write(f"**Match Percentage:** {match_percentage}%")
    st.write("**Extracted Resume Skills:**", resume_skills)
    st.write("**Extracted Job Skills:**", job_skills)

    st.subheader("Suggestions")
    st.write("**Additional Skills to Add:**")
    st.write(additional_skills)
    st.write("**Additional Skills to remove:**")
    st.write(additional_remskills)
    # Visualization
    st.subheader("Skill Match Visualization")
    plot_skill_match(resume_skills, job_skills)

    # Chatbot Feature
    st.subheader("Ask Your Resume")
    chat_input = st.text_input("Ask a question about your resume (e.g., 'How can I improve my resume?')")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = None

    if chat_input:
        with st.spinner("Thinking..."):
            chatbot_reply, st.session_state["chat_history"] = chatbot_response(chat_input, st.session_state["chat_history"])
        st.write("**Chatbot Response:**")
        st.write(chatbot_reply)

else:
    st.warning("Please upload a resume and provide a job description.")