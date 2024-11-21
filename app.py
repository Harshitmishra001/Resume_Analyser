import os
import json
import csv
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from dotenv import load_dotenv
from flask_cors import CORS
import uuid
import PyPDF2
import re

load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-pro")
model = genai.GenerativeModel("gemini-1.5-pro")

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

# Extract text from the resume PDF
def extract_text_from_pdf():
    file_path = r'uploads/data.pdf'
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

# Add interviewee data to CSV
def add_interviewee_to_csv(interviewee_data, filename="interviewee.csv"):
    header = ["name", "skills", "total_experience_years"]
    next_serial_no = 1
    existing_data = set()

    if os.path.exists(filename):
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip header row
            for row in reader:
                next_serial_no += 1
                existing_data.add(tuple(row[1:]))  # Store data as tuples (excluding serial no.)

    # Check if interviewee data already exists
    if tuple(interviewee_data) in existing_data:
        print("Data already exists. Skipping...")
        return

    # Append new data if not a duplicate
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if next_serial_no == 1:  # Add header if file is new
            writer.writerow(header)
        interviewee_data.insert(0, next_serial_no)
        writer.writerow(interviewee_data)

# Extract details from resume text
def extract_details(text):
    prompt = """
    You are a helpful AI assistant. Your task is to extract details from the provided resume text. Respond in the following format:
    ```json
    [
        "name", 
        "skills",
        "total_experience_years", 
    ]
    ```
    Where:
    -  If the information cannot be found, use "NA".
    -  "total_experience_years" should be an integer representing the total years of experience. If it's not mentioned, use "NA".
    - "skills" list all the technical skills that are present in the resume.
    ```
    {text}
    ```
    """
    try:
        response = model.generate_content([prompt.format(text=text)])
        extracted_data = response.text

        try:
            actual_list = json.loads(extracted_data)
        except json.JSONDecodeError:
            pattern = r"(.*?)" 
            match = re.search(pattern, extracted_data, re.DOTALL)
            if match:
                items = [item.strip().replace('"', '') for item in match.group(1).split(",")]
                actual_list = [int(item) if item.isdigit() else item for item in items]
            else:
                actual_list = ["Error: Unable to extract data"]
        return actual_list
    except Exception as e:
        print("Error extracting details:", e)
        return None

# Function to calculate match percentage and missing skills
def compare_skills(resume_skills, job_skills):
    common_skills = set(resume_skills).intersection(set(job_skills))
    missing_skills = set(job_skills) - set(resume_skills)
    match_percentage = (len(common_skills) / len(job_skills)) * 100
    return match_percentage, list(missing_skills)

# Load job skills from skills.json
def load_job_skills(job_type):
    try:
        with open('skills.json', 'r') as f:
            job_skills = json.load(f)
            return job_skills.get(job_type, [])
    except FileNotFoundError:
        print("Error: skills.json not found.")
        return []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        filename = "data.pdf"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        data = extract_text_from_pdf()
        resume_details = extract_details(data)

        # Check if resume_details is properly populated
        print(resume_details)  # Debugging
        
        # Safely access details
        if resume_details and len(resume_details) > 1:
            resume_skills = resume_details[1]
        else:
            resume_skills = []  # Default to an empty list if skills are not found

        # Add extracted details to CSV or process further
        add_interviewee_to_csv(resume_details)
        
        return render_template('index.html', details=resume_details, skills=resume_skills)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)