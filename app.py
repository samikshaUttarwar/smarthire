
"""
SmartHire — Streamlit App (with file upload)
Run with: streamlit run app.py
"""

import streamlit as st
from nlp_engine import full_analysis
from file_reader import extract_text_from_file

st.set_page_config(page_title="SmartHire", page_icon="🔍", layout="centered")

st.title("SmartHire")
st.caption("AI resume screener — upload your resume or paste text, then add a job description to get an instant fit analysis.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Your resume**")
    resume_file = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        key="resume_upload"
    )

    resume_text = ""
    if resume_file is not None:
        try:
            resume_text = extract_text_from_file(resume_file)
            st.success(f"Loaded {resume_file.name} ({len(resume_text)} characters)")
        except Exception as e:
            st.error(f"Couldn't read this file: {e}")

    resume_text = st.text_area(
        "Or paste resume text",
        value=resume_text,
        height=200,
        placeholder="Paste resume text here if you didn't upload a file..."
    )

with col2:
    st.markdown("**Job description**")
    jd = st.text_area(
        "Paste the job description",
        height=260,
        placeholder="Paste the job description you're applying for..."
    )

analyze = st.button("Analyze fit", type="primary", use_container_width=True)

if analyze:
    if not resume_text.strip() or not jd.strip():
        st.warning("Upload or paste a resume, and paste a job description, to continue.")
    else:
        with st.spinner("Analyzing... (first run downloads the AI model, ~1.6 GB)"):
            result = full_analysis(resume_text, jd)

        st.divider()

        c1, c2, c3 = st.columns(3)
        c1.metric("Fit score", f"{result['fit_score']}%")
        c2.metric("Skills matched", f"{len(result['skills_matched'])}")
        c3.metric("Skill gaps", f"{len(result['skill_gaps'])}")

        st.progress(min(int(result["fit_score"]), 100) / 100)

        st.subheader("Skills matched")
        st.write(", ".join(result["skills_matched"]) or "No direct matches found.")

        st.subheader("Skill gaps")
        if result["skill_gaps"]:
            st.write(", ".join(result["skill_gaps"]))
        else:
            st.success("No gaps — great match!")

        st.subheader("Bonus skills you have")
        st.write(", ".join(result["bonus_skills"]) or "None identified beyond the JD's requirements.")

        st.subheader("Recommendations")
        for r in result["recommendations"]:
            st.write(f"- {r}")

st.divider()
st.caption("Built with Python, spaCy, NLTK, Hugging Face Transformers, scikit-learn, and pdfplumber.")
