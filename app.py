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
    # DRDO_Department=["Advanced Systems Laboratory (ASL)","Aerial Delivery Research and Development Establishment (ADRDE)","Aeronautical Development Establishment (ADE)","Armament Research & Development Establishment (ARDE)",
    #                 "Centre For Air Borne System (CABS)","Centre for Artificial Intelligence & Robotics (CAIR)","Centre for Fire, Explosive and Environment Safety (CFEES)","Combat Vehicles Research & Development Establishment (CVRDE)","Defence Bio-Engineering & Electro Medical Laboratory (DEBEL)",
    #                 "Defence Electronics Application Laboratory (DEAL)","Defence Electronics Research Laboratory (DLRL)","Defence Food Research Laboratory (DFRL)","Defence Geoinformatics Research Establishment (DGRE)","Defence Institute of High Altitude Research (DIHAR)","Defence Institute of Physiology & Allied Sciences (DIPAS)",
    #                 "Defence Institute of Psychological Research (DIPR)","Defence Laboratory Jodhpur (DLJ)","Defence Materials and Stores Research and Development Establishment (DMSRDE)","Defence Metallurgical Research Laboratory (DMRL)","Defence Research & Development Laboratory (DRDL)","Defence Research Development Establishment (DRDE)","Defence Research Laboratory (DRL)","Defence Scientific Information & Documentation Centre (DESIDOC)","Defense young Scientist Laboratory - Artificial Intelligence (DYSL-AI)","DRDO Young Scientist Laboratory (DYSL-AT)",
    #                 "DRDO Young Scientist Laboratory (DYSL-CT)","DRDO Young Scientist Laboratory (DYSL-QT)","DRDO Young Scientist Laboratory (DYSL-SM)","Electronics & Radar Development Establishment (LRDE)","Gas Turbine Research Establishment (GTRE)","High Energy Materials Research Laboratory (HEMRL)","Institute for Systems Studies & Analyses (ISSA)","Institute of Nuclear Medicine & Allied Sciences (INMAS)",
    #                 "Institute of Technology Management (ITM)","Instruments Research & Development Establishment (IRDE)","Integrated Test Range (ITR)","Naval Materials Research Laboratory (NMRL)","Naval Physical & Oceanographic Laboratory (NPOL)","Naval Science & Technological Laboratory (NSTL)","Proof & Experimental Establishment (PXE)","Research & Development Establishment (Engrs.)","Research Centre Imarat (RCI)","Scientific Analysis Group (SAG)","Solid State Physics Laboratory (SSPL)","Terminal Ballistics Research Laboratory (TBRL)","Vehicles Research and Development Establishment (VRDE)"]
    prompt = """
    You are a helpful AI assistant. Your task is to extract details from the provided resume text. Respond in the following format:
    ```json
    [
        "name", 
        "Qualifications", 
        "What field Reseach Papers are on", 
        "total_experience_years", 
        "Relevant_Department" 
    ]
    ```
    Where:
    -  If the information cannot be found, use "NA".
    -  "total_experience_years" should be an integer representing the total years of experience. If it's not mentioned, use "NA".
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
        print(extract_details(data))
        print(extract_text_from_pdf())
        return render_template('index.html', details=extract_details(data))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)