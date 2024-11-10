from flask import Flask, request, render_template
import os
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Text extraction functions for different file formats
EXTRACTORS = {
    '.pdf': lambda path: ''.join([page.extract_text() for page in PyPDF2.PdfReader(path).pages]),
    '.docx': docx2txt.process,
    '.txt': lambda path: open(path, 'r', encoding='utf-8').read()
}

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    return EXTRACTORS.get(ext, lambda path: "")(file_path)

@app.route("/")
def matchresume():
    return render_template('matchresume.html')

@app.route('/matcher', methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_files = request.files.getlist('resumes')

        if not job_description or not resume_files:
            return render_template('matchresume.html', message="Please upload resumes and enter a job description.")

        # Extract text from resumes
        resumes = [extract_text(os.path.join(app.config['UPLOAD_FOLDER'], resume.filename)) for resume in resume_files]
        
        # Vectorize job description and resumes
        vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
        similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()

        # Get top 5 resumes
        top_indices = similarities.argsort()[-5:][::-1]
        top_resumes = [(resume_files[i].filename, round(similarities[i], 2)) for i in top_indices]

        return render_template('matchresume.html', message="Top matching resumes:", top_resumes=top_resumes)

    return render_template('matchresume.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
