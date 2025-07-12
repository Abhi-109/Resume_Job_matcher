import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from utils.extract_resume import extract_text

# ---------------------- Setup ----------------------
st.set_page_config(page_title="Resume Matcher", page_icon="📄", layout="wide")

# Custom CSS for a clean, modern look
st.markdown("""
    <style>
    .main {background-color: #1e1e1e; color: #e0e0e0;}
    .sidebar .sidebar-content {background-color: #2c2c2c;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
    .job-card {background-color: #2f2f2f; padding: 15px; border-radius: 8px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);}
    .job-title {color: #e0e0e0; font-size: 1.2em;}
    .job-detail {color: #bbb; margin: 5px 0;}
    .job-link {color: #4CAF50; text-decoration: none;}
    .score {color: #ff69b4; font-weight: bold;}
    .uploaded-file {color: #ff69b4; font-weight: bold; margin-bottom: 10px;}
    </style>
""", unsafe_allow_html=True)

# ---------------------- Load Model & Job Data ----------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_jobs():
    df = pd.read_csv("D:\Coding\Resume_matcher_project\Data\merged_jobs_light.csv")
    return df.dropna(subset=["job_text"]).reset_index(drop=True)

model = load_model()
job_df = load_jobs()

# ---------------------- Main UI ----------------------
st.title("🔍 Resume Matcher")
st.markdown("Upload your resume (PDF/DOCX) to find the best matching jobs using BERT-based matching.")

# Sidebar
with st.sidebar:
    st.header("Upload")
    uploaded_file = st.file_uploader("📄 Choose a file", type=["pdf", "docx"])
    if uploaded_file:
        st.markdown(f'<div class="uploaded-file">📄 {uploaded_file.name}</div>', unsafe_allow_html=True)
    if st.button("Clear", key="clear"):
        st.session_state.uploaded_file = None
        st.rerun()

# Process and display results
if uploaded_file:
    with st.spinner("Matching your resume..."):
        file_type = uploaded_file.name.split('.')[-1].lower()
        resume_text = extract_text(uploaded_file, file_type)

        if resume_text:
            # Generate embeddings
            resume_embedding = model.encode(resume_text, convert_to_tensor=True)
            job_embeddings = model.encode(job_df["job_text"].tolist(), convert_to_tensor=True, batch_size=32)
            
            # Compute similarity
            similarity_scores = util.cos_sim(resume_embedding, job_embeddings)[0].cpu().numpy()
            top_indices = np.argsort(similarity_scores)[::-1][:5]

            # Display top 5 jobs
            st.subheader("🎯 Top 5 Matching Jobs")
            for idx in top_indices:
                job = job_df.iloc[idx]
                score = round(similarity_scores[idx] * 100, 2)
                st.markdown(f"""
                    <div class="job-card">
                        <div class="job-title">💼 {job['Job Title']}</div>
                        <div class="job-detail">🏢 {job.get('Company', 'Unknown')}</div>
                        <div class="job-detail">📍 {job.get('location', 'N/A')}</div>
                        <div class="job-detail"><a href="{job.get('link', '#')}" class="job-link">🔗 View Job</a></div>
                        <div class="score">🧠 Similarity: {score}%</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Could not extract text from the resume.")
else:
    st.info("Please upload a resume to get started.")
