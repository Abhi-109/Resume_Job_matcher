import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import custom utility functions
from utils.extract_resume import extract_text, extract_keywords

# ----------------- Page Setup ----------------- #
st.set_page_config(page_title="Resume Matcher", layout="wide")
st.title("📄 Resume Matcher & Job Recommender")
st.markdown("Upload your resume and get matched with the most relevant jobs based on your skills.")

# ----------------- Load Job Dataset ----------------- #
@st.cache_data
def load_jobs():
    df = pd.read_csv(r"D:\Coding\Resume_matcher_project\Data\merged_jobs.csv")
    df = df.dropna(subset=['job_text'])
    df['job_text'] = df['job_text'].astype(str)

    return df

job_df = load_jobs()

# ----------------- Resume Upload Section ----------------- #
st.sidebar.header("📁 Upload Resume")
uploaded_file = st.sidebar.file_uploader("Upload a Resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()
    resume_text = extract_text(uploaded_file, file_type)

    if resume_text:
        st.subheader("📄 Resume Text Preview (first 500 characters):")
        st.text(resume_text[:500])

        resume_keywords = extract_keywords(resume_text, num_keywords=30)
        st.subheader("🔑 Extracted Keywords from Resume:")
        st.write(resume_keywords)

        resume_string = ' '.join(resume_keywords)

        # ----------------- Match Resume with Job Descriptions ----------------- #
        all_text = pd.concat([pd.Series([resume_string]), job_df['job_text']], ignore_index=True)

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(all_text)

        resume_vector = tfidf_matrix[0]
        job_vectors = tfidf_matrix[1:]

        similarity_scores = cosine_similarity(resume_vector, job_vectors).flatten()
        top_indices = similarity_scores.argsort()[::-1][:5]

        st.subheader("🎯 Top 5 Matching Jobs:")
        for idx in top_indices:
            job = job_df.iloc[idx]
            st.markdown(f"""
            🔹 **{job['job_title']}**  
            🏢 *{job['company_name']}*  
            📍 {job['company_address_locality']}, {job['company_address_region']}  
            🔗 [Company Website]({job['company_website']})  
            ✅ **Match Score:** {round(similarity_scores[idx] * 100, 2)}%
            """)
