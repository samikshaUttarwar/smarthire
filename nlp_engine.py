"""
SmartHire - NLP Engine
Handles text cleaning, skill extraction, fit scoring, and recommendations.
"""

import re
import nltk
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

nlp = spacy.load("en_core_web_sm")

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-3"
        )
    return _classifier


SKILL_LABELS = [
    "Python", "JavaScript", "React", "Node.js", "Express.js",
    "MongoDB", "SQL", "Java", "C++", "Machine Learning",
    "Deep Learning", "NLP", "TensorFlow", "PyTorch", "AWS",
    "Docker", "Kubernetes", "Git", "REST API", "GraphQL",
    "HTML", "CSS", "TypeScript", "Data Structures", "Algorithms",
    "Agile", "CI/CD", "Linux", "Firebase", "Redis"
]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords(text):
    doc = nlp(clean_text(text))
    stop = set(stopwords.words("english"))
    keywords = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and len(token.text) > 2
        and token.lemma_ not in stop
    ]
    return list(set(keywords))


def extract_skills(text, labels=None):
    labels = labels or SKILL_LABELS
    classifier = _get_classifier()
    result = classifier(text[:1000], candidate_labels=labels, multi_label=True)
    return [
        label for label, score in zip(result["labels"], result["scores"])
        if score > 0.5
    ]


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
