import os
import re
import json 
import csv
from flask import Flask, request, jsonify,render_template
import fitz 
import google.generativeai as genai
from dotenv import load_dotenv
from flask_cors import CORS 
import uuid
import PyPDF2

load_dotenv()  
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-pro") 

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)
def extract_text_from_pdf():
    file_path=r'uploads\data.pdf'
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text
def add_interviewee_to_csv(interviewee_data, filename="interviewee.csv"):
    header = ["serial no.", "name", "Qualifications", "Research_field", "total_experience_years", "Relevant_Department"]
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
def extract_details(text):
    prompt = """
    You are a helpful AI assistant. Your task is to extract details from the provided resume text. Respond in the following format:
    ```json
    [
        "name", 
        "Qualifications", 
        "Research_field", 
        "total_experience_years", 
        "Relevant_Department" 
    ]
    ```
    Where:
    -  If the information cannot be found, use "NA".
    -  "total_experience_years" should be an integer representing the total years of experience. If it's not mentioned, use "NA".
    - "Qualification" should be the highest qualification for example if a person has done BTech, Mtech and PHD then store word PHD only.
    -  "Research_field" is where you will read the mulitiple topics of reaserch paper and store overall 1 field in which his research papers are for.
    - "Relevant_Department" should be from the the various departments under Defence Research and Development Organisation Laboratory and Establishments of INDIA, select the one best for the resume.If none suitable , use "NA"
    Resume Text:
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
            pattern = r"\[(.*?)\]" 
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



@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        file = request.files['file']
        filename = "data.pdf"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        data=extract_text_from_pdf()
        details=extract_details(data)
        print(details)
        add_interviewee_to_csv(details)
        return render_template('index.html', details=details)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)