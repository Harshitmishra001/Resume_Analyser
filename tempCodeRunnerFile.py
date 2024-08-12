@app.route('/analyze', methods=['POST'])
# def analyze_resume():
#     try:
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file part'}), 400

#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({'error': 'No selected file'}), 400

#         if file:
#             text = extract_text_from_pdf(file.read())
#             if text is None: 
#                 return jsonify({'error': 'Error extracting text from PDF'}), 500 

#             extracted_data = extract_details(text)
#             if extracted_data is None:
#                 return jsonify({'error': 'Error extracting details from text'}), 500

#             return jsonify(extracted_data), 200

#     except Exception as e:
#         print("Error during analysis:", e)
#         return jsonify({'error': 'An error occurred during processing'}), 500
