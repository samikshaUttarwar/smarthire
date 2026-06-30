"""
SmartHire - NLP Engine (fast version, no heavy AI model)
Handles text cleaning, skill extraction, fit scoring, and recommendations.
"""

import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


SKILL_LABELS = [
    "Python", "JavaScript", "React", "React.js", "Node.js", "Node",
    "Express.js", "Express", "MongoDB", "SQL", "Java", "C++",
    "Machine Learning", "ML", "Deep Learning", "NLP",
    "TensorFlow", "PyTorch", "AWS", "Docker", "Kubernetes",
    "Git", "GitHub", "REST API", "REST", "API", "GraphQL",
    "HTML", "CSS", "TypeScript", "Data Structures", "DSA",
    "Algorithms", "Agile", "CI/CD", "Linux", "Firebase", "Redis",
    "JWT", "bcrypt", "OOP", "Postman", "Streamlit", "Pandas",
    "NumPy", "scikit-learn", "Hugging Face", "LangChain",
    "Prompt Engineering", "Groq", "LLaMA", "CST Studio",
    "5G", "4G", "RF", "Microwave", "Antenna", "VSWR"
]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords(text):
    stop = set(stopwords.words("english"))
    cleaned = clean_text(text)
    words = cleaned.split()
    keywords = [w for w in words if w not in stop and len(w) > 2]
    return list(set(keywords))


def extract_skills(text, labels=None):
    labels = labels or SKILL_LABELS
    cleaned = clean_text(text)
    found = []
    for label in labels:
        label_clean = clean_text(label)
        pattern = r"\b" + re.escape(label_clean).replace(r"\ ", r"[\s\-]?") + r"\b"
        if re.search(pattern, cleaned):
            found.append(label)
    return found


def calculate_fit_score(resume_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([clean_text(resume_text), clean_text(jd_text)])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 1)


def analyze_skills(resume_text, jd_text):
    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(jd_text))

    matched = resume_skills & jd_skills
    gaps = jd_skills - resume_skills
    bonus = resume_skills - jd_skills

    match_pct = round(len(matched) / max(len(jd_skills), 1) * 100)

    return {
        "matched": sorted(matched),
        "gaps": sorted(gaps),
        "bonus": sorted(bonus),
        "match_pct": match_pct,
    }


def get_recommendations(score, gaps, matched):
    recs = []

    if score < 50:
        recs.append("Your resume needs stronger keyword alignment with this job description.")
    elif score < 70:
        recs.append("Decent overlap - tighten your summary to mirror the JD's exact phrasing.")
    else:
        recs.append("Strong match. Lead your resume summary with your top 3 relevant projects.")

    if len(gaps) > 3:
        recs.append("Add these missing skills if you have them: " + ", ".join(list(gaps)[:3]) + ".")
    elif len(gaps) > 0:
        recs.append("Consider addressing this skill gap: " + ", ".join(gaps) + ".")

    if len(matched) == 0:
        recs.append("No direct skill overlap found - this JD may not be the best fit, or your resume needs rework.")

    recs.append("Quantify achievements with numbers wherever possible.")
    recs.append("Make sure your most relevant experience appears in the first half of the resume.")

    return recs


def full_analysis(resume_text, jd_text):
    score = calculate_fit_score(resume_text, jd_text)
    skills = analyze_skills(resume_text, jd_text)
    recos = get_recommendations(score, skills["gaps"], skills["matched"])

    return {
        "fit_score": score,
        "skills_matched": skills["matched"],
        "skill_gaps": skills["gaps"],
        "bonus_skills": skills["bonus"],
        "match_pct": skills["match_pct"],
        "recommendations": recos,
    }
