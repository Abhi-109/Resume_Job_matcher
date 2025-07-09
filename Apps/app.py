import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.extract_resume import extract_text, extract_keywords

# ---------------------- Setup ----------------------
st.set_page_config(page_title="Resume Matcher", page_icon="📄", layout="wide")
st.title("🔍 Resume Matcher")
st.markdown("Upload your resume and get matched with the most relevant jobs based on your skills.")

# ---------------------- Load Job Data ----------------------
@st.cache_data
def load_jobs():
    df = pd.read_csv("D:\Coding\Resume_matcher_project\Data\merged_jobs_light.csv")  # <-- make sure this path is correct
    df = df.dropna(subset=["job_text"])  # job_text is needed for matching
    df.reset_index(drop=True, inplace=True)
    return df

job_df = load_jobs()

# ---------------------- Upload Resume ----------------------
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📄 Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()
    resume_text = extract_text(uploaded_file, file_type)

    if resume_text:
        # Extract keywords and convert to a string
        keywords = extract_keywords(resume_text, num_keywords=30)
        resume_keywords_str = ' '.join(keywords)

        # ---------------- TF-IDF Vectorization ----------------
        all_text = pd.concat([pd.Series([resume_keywords_str]), job_df['job_text']], ignore_index=True)
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(all_text)

        resume_vector = tfidf_matrix[0]
        job_vectors = tfidf_matrix[1:]

        # ---------------- Cosine Similarity ----------------
        similarity_scores = cosine_similarity(resume_vector, job_vectors).flatten()
        top_indices = similarity_scores.argsort()[::-1][:5]

        # ---------------- Display Results ----------------
        st.subheader("🎯 Top 5 Matching Jobs")
        for idx in top_indices:
            job = job_df.iloc[idx]
            st.markdown(f"""
            ### 💼 {job['job_title']}
            🏢 **{job.get('company', 'Unknown')}**  
            📍 {job.get('job_location', 'Not specified')}  
            🔗 [View Job Posting]({job.get('job_link', '#')})  
            🧠 **Similarity Score:** {round(similarity_scores[idx]*100, 2)}%
            ---
            """)
    else:
        st.warning("Couldn't extract text from the uploaded resume.")
