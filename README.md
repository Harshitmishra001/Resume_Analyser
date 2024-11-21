Resume and Job Description Matcher

A Python-based application that leverages Generative AI to analyze resumes and job descriptions. This tool calculates match percentages, identifies gaps, and provides actionable suggestions to improve resumes for specific job requirements.


---

Features

1. PDF Parsing
Extract text from uploaded resume PDFs.


2. Skill Extraction
Automatically identify skills from resumes and job descriptions using Natural Language Processing (NLP).


3. Match Percentage Calculation
Compare the skills from the resume and job description to calculate a match percentage.


4. Skill Suggestions
Discover missing skills in the resume and get recommendations to align with the job description.


5. Generative AI Rephrasing
Rephrase the resume content and job description for improved clarity and alignment.


6. Chatbot Interaction
Chat with your resume to ask questions and receive tailored recommendations.


7. Improvement Suggestions
Get actionable advice on how to enhance your resume based on the job description.




---

Installation

Step 1: Clone the Repository

git clone https://github.com/Harshitmishra001/Resume_Analyser.git
cd  Resume_Analyser

Step 2: Install Dependencies

Install the required libraries using pip:

pip install pdfplumber spacy scikit-learn transformers streamlit

Step 3: Download spaCy Model

Download the English NLP model for spaCy:

python -m spacy download en_core_web_sm


---

Usage

Run the Application

Start the Streamlit app:

streamlit run app2.py

Access the App

Open your browser and navigate to the URL provided by Streamlit (typically http://localhost:8501).

How to Use

1. Upload your resume (PDF format).


2. Enter the job description in the provided text area.


3. View:

Match Percentage

Extracted Skills (from both resume and job description)

Skill Suggestions

Improvement Recommendations



4. Use the Chatbot to ask questions like:

"What skills am I missing?"

"How can I improve my resume?"





---

Technologies Used

Python: Core programming language.

Streamlit: Framework for building user-friendly web interfaces.

spaCy: NLP library to extract skills and analyze text.

Transformers (Hugging Face): Generative AI for rephrasing and chatbot functionalities.

scikit-learn: For calculating match percentages using cosine similarity.

pdfplumber: Extract text content from resume PDFs.



---

Features in Action

1. Match Percentage Calculation

Goal: Understand how well your resume aligns with the job description.

How: Compare extracted skills and calculate a cosine similarity percentage.


2. Skill Suggestions

Goal: Add missing skills to your resume to meet job requirements.

How: Identify skills present in the job description but missing in your resume.


3. Chatbot Interaction

Goal: Receive dynamic suggestions through a conversational interface.

Examples:

"What are the weak points in my resume?"

"How can I tailor my resume for this job?"




---

Future Improvements

AI-Generated Resume Sections: Automatically create resume sections like "Objective" or "Experience".

Advanced Chatbot: Incorporate a more sophisticated AI model for richer interactions.

Mobile Support: Make the app mobile-friendly for on-the-go users.



---

License

This project is licensed under the MIT License. See the LICENSE file for more details.


---

Contributing

We welcome contributions! To contribute:

1. Fork this repository.


2. Create a feature branch (git checkout -b feature-name).


3. Commit your changes (git commit -m "Add feature").


4. Push your branch (git push origin feature-name).


5. Submit a pull request.




---

Contact

For questions, feedback, or support:

Name: Harshit mishra

Email: hmharsh123@gmail.com

GitHub: harshitmishra001



---

Let me know if you'd like me to customize any specific section or add placeholders for the screenshots!
