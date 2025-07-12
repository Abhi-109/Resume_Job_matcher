
# Resume & Job Matcher

A smart job recommendation system built using **Sentence-BERT**, **Streamlit**, and **Pandas**. Upload your resume (PDF or DOCX), and get matched with the most relevant jobs based on your skills, experience, and profile content.

**Live Demo**: [Click here to try it now!](https://resumejobmatcher-byabhi-109.streamlit.app/)

---

## Features

- Upload resumes in PDF or DOCX format  
- Extracts text and keywords using NLP  
- Matches resumes with job descriptions using **BERT-based semantic similarity**  
- Displays **Top 5 most relevant job recommendations** with similarity scores  
- Fast and intuitive UI powered by **Streamlit**

---

## How It Works

1. **Resume Parsing**  
   - Uses `PyMuPDF` and `python-docx` to extract text from resumes  
   - Cleans and preprocesses the content for keyword extraction  

2. **Job Dataset**  
   - Contains fields: `Job Title`, `Job Description`, `Skills`, `Company`, `Location`  
   - Combines fields into a `job_text` column for better contextual matching  

3. **Semantic Matching**  
   - Embeds resumes and job descriptions using `sentence-transformers`  
   - Computes **cosine similarity** to find the most relevant job matches  

---

## Project Structure

```
resume_matcher_project/
├── Apps/
│   ├── app.py
│   └── utils/
│       └── extract_resume.py
├── Data/
│   ├── jobs.csv
│   └── resumes.csv
├── requirements.txt
└── README.md
```

---

##  Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/resume-job-matcher.git
cd resume-job-matcher
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run Apps/app.py
```

---

## Tech Stack

- `streamlit`
- `sentence-transformers`
- `pandas`
- `scikit-learn`
- `python-docx`
- `PyMuPDF`

---

## Sample Output

```
🎯 Top 5 Matching Jobs
💼 Data Scientist
🏢 Google
📍 Bangalore, India
🧠 Similarity Score: 89.52%
```

---

## Contributions Welcome

- Open issues for bugs or feature requests  
- Fork the repo and raise a pull request 🚀  

---

## Live Demo

👉 [https://resumejobmatcher-byabhi-109.streamlit.app](https://resumejobmatcher-byabhi-109.streamlit.app)

---

## Future Ideas

- Add filters for job types (Remote / Onsite / Hybrid)  
- AI-powered resume improvement suggestions  
- Better section-wise parsing (Skills, Experience, Education)  
- Job application tracker integration  

---

## License

**MIT License** © 2025 Abhishek
