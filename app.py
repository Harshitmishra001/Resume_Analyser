import os
import re
import json 
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

def extract_details(text):
    prompt = """
    You are a helpful AI assistant. Your task is to extract details from the provided resume text. Respond in the following format:
    ```json
    [
        "name", 
        "related to which field", 
        "birthplace", 
        "total_experience_years", 
        "current_or_most_recent_job_title" 
    ]
    ```
    Where:
    -  If the information cannot be found, use "NA".
    -  "total_experience_years" should be an integer representing the total years of experience. If it's not mentioned, use "NA".
    
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
        print(extract_details(data))
        return render_template('index.html', details=extract_details(data))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)