
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
from streamlit_mic_recorder import mic_recorder
import whisper
import io
import os
from gtts import gTTS
import base64
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import csv
from datetime import datetime
import google.generativeai as genai
import json
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Python Voice Quiz", layout="centered")
st.title("🎤 Python Voice Quiz (Strict Evaluation)")

st.session_state.setdefault('selected_level', None)
st.session_state.setdefault('user_info_collected', False)
st.session_state.setdefault('user_name', "")
st.session_state.setdefault('questions', [])
st.session_state.setdefault('current_q', 0)
st.session_state.setdefault('answers', [])
st.session_state.setdefault('submitted', False)
st.session_state.setdefault('evaluations', [])
st.session_state.setdefault('question_metadatas', [])
st.session_state.setdefault('audio_played', False)
st.session_state.setdefault('results_audio_generated', False)
st.session_state.setdefault('gemini_feedback', "")
USER_DATA_FILE = "user_data.csv"

# ---------------------- DATA SAVE FUNCTIONS ---------------------
def save_user_data(name, email, phone):
    file_exists = os.path.isfile(USER_DATA_FILE)
    with open(USER_DATA_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Name", "Email", "Phone"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, email, phone])

# ---------------------- AUDIO ---------------------
def autoplay_audio(audio_bytes):
    audio_str = "data:audio/wav;base64,%s" % (base64.b64encode(audio_bytes).decode())
    audio_html = f"""
    <audio autoplay>
        <source src="{audio_str}" type="audio/wav">
    </audio>
    """
    st.components.v1.html(audio_html, height=0)

def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang, slow=False)
    audio_file = io.BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    return audio_file.read()

# ---------------------- TABS ---------------------
def safe_show_tabs(evaluations, questions, answers):
    if not evaluations:
        st.warning("No evaluations available. Please ensure you answered all questions.")
        return
    tabs = st.tabs([f"Q{i+1}" for i in range(len(evaluations))])
    for i, tab in enumerate(tabs):
        with tab:
            eval_data = evaluations[i]
            st.markdown(f"**Question {i+1}:** {questions[i]}")
            st.code(answers[i])
            st.markdown(f"**Score:** {eval_data['score']}/5")
            st.progress(eval_data['score'] / 5)
            st.markdown("**Details:**")
            st.write(eval_data['justification'])
            st.markdown("**Reference Answer:**")
            st.info(eval_data['reference_answer'])

@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path="python_quiz_db")
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection(
        name="python_questions",
        embedding_function=embedding_function
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    langchain_chroma = Chroma(
        client=client,
        collection_name="python_questions",
        embedding_function=embeddings
    )
    return collection, langchain_chroma

chroma_collection, langchain_chroma = get_chroma_collection()
model = SentenceTransformer("all-MiniLM-L6-v2")  # Fixed model name

def keyword_match(user_answer: str, reference_answer: str) -> int:
    ref_words = set(re.findall(r'\b\w+\b', reference_answer.lower()))
    user_words = set(re.findall(r'\b\w+\b', user_answer.lower()))
    return len(ref_words & user_words)

def strict_evaluation(question: str, user_answer: str) -> dict:
    if not user_answer.strip():
        return {"score": 0, "justification": "Empty answer provided", "reference_answer": ""}

    try:
        result = chroma_collection.query(query_texts=[question], n_results=1)
        ref_answer = result["documents"][0][0]
    except:
        ref_answer = "Reference answer not found."

    emb_user = model.encode([user_answer])[0].reshape(1, -1)
    emb_ref  = model.encode([ref_answer])[0].reshape(1, -1)
    similarity = float(cosine_similarity(emb_user, emb_ref)[0][0] * 100)
    keywords_matched = keyword_match(user_answer, ref_answer)

    if similarity >= 50 and keywords_matched >= 6:
        score = 5
    elif similarity >= 40 and keywords_matched >= 4:
        score = 4
    elif similarity >= 30 and keywords_matched >= 3:
        score = 3
    elif similarity >= 25 and keywords_matched >= 2:
        score = 2
    elif similarity >= 15 and keywords_matched >= 1:
        score = 1
    else:
        score = 0

    return {
        "score": score,
        "justification": f"Similarity: {similarity:.2f}%\nKeywords matched: {keywords_matched}",
        "reference_answer": ref_answer
    }

def generate_final_feedback(evaluations, questions, answers):
    """Generate comprehensive feedback using Gemini"""
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Prepare evaluation summary
        summary = "Evaluation Summary:\n"
        for i, eval_data in enumerate(evaluations):
            summary += f"\nQuestion {i+1}: {questions[i]}\n"
            summary += f"Your Answer: {answers[i]}\n"
            summary += f"Score: {eval_data['score']}/5\n"
            summary += f"Feedback: {eval_data['justification']}\n"
        
        prompt = f"""You are a Python learning assistant. Provide constructive feedback based on this quiz performance:
        
        {summary}
        
        Please provide:
        1. Overall assessment of the performance
        2. Key strengths demonstrated
        3. Areas needing improvement
        4. Suggested learning resources or topics to focus on
        5. Encouraging closing remarks
        
        Keep it professional yet friendly, and limit to 3-4 paragraphs.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not generate feedback: {str(e)}"

# ------------------------- USER INFO COLLECTION ---------------------
if not st.session_state.user_info_collected:
    with st.form("user_info_form"):
        st.subheader("Please Enter Your Details")
        name = st.text_input("Full Name*", placeholder="Enter your full name")
        email = st.text_input("Email*", placeholder="Enter your email")
        phone = st.text_input("Phone Number*", placeholder="Enter your phone number")

        submitted = st.form_submit_button("Submit")
        if submitted:
            if name and email and phone:
                save_user_data(name, email, phone)
                st.session_state.user_info_collected = True
                st.session_state.user_name = name
                st.rerun()
            else:
                st.error("Please fill in all required fields (marked with *)")

# ------------------------- DIFFICULTY SELECTION ---------------------
if st.session_state.user_info_collected and st.session_state.selected_level is None:
    st.subheader("Select Difficulty Level:")
    col1, col2, col3 = st.columns(3)
    if col1.button("🟢 Beginner"):
        st.session_state.selected_level = "Beginner"
    if col2.button("🟠 Intermediate"):
        st.session_state.selected_level = "Intermediate"
    if col3.button("🔴 Advanced"):
        st.session_state.selected_level = "Advanced"

if st.session_state.selected_level:
    level_colors = {"Beginner": "🟢", "Intermediate": "🟠", "Advanced": "🔴"}
    st.markdown(f"### Selected Difficulty: {level_colors[st.session_state.selected_level]} {st.session_state.selected_level}")

# ------------------------- GET QUESTIONS ---------------------
def get_questions_by_difficulty(level, n=10):
    try:
        results = chroma_collection.query(
            query_texts=[f"Python {level} level questions"],
            n_results=20,
            where={"difficulty": level}
        )
        import random
        combined = list(zip(results['documents'][0], results['metadatas'][0]))
        random.shuffle(combined)
        selected = combined[:n]
        questions = [q[0] for q in selected]
        metadatas = [q[1] for q in selected]
        return questions, metadatas
    except Exception as e:
        st.error(f"Error fetching questions: {str(e)}")
        return [], []

# ------------------------- QUIZ ---------------------
if st.session_state.selected_level and not st.session_state.questions:
    if st.button("🎤 Start Voice Quiz"):
        questions, metadatas = get_questions_by_difficulty(st.session_state.selected_level, 5)
        if questions:
            st.session_state.questions = questions
            st.session_state.question_metadatas = metadatas
            st.session_state.current_q = 0
            st.session_state.answers = [''] * len(questions)
            st.session_state.submitted = False
            st.session_state.audio_played = False
            st.session_state.evaluations = []
            st.session_state.results_audio_generated = False
            st.rerun()

if st.session_state.questions and not st.session_state.submitted:
    q_index = st.session_state.current_q

    if not st.session_state.audio_played:
        question_text = f"Question {q_index + 1}. {st.session_state.questions[q_index]}"
        audio_bytes = text_to_speech(question_text)
        autoplay_audio(audio_bytes)
        st.session_state.audio_played = True

    st.subheader(f"Question {q_index + 1} of {len(st.session_state.questions)}")
    difficulty = st.session_state.question_metadatas[q_index]['difficulty']
    badge = "🟢 Beginner" if difficulty == "Beginner" else "🟠 Intermediate" if difficulty == "Intermediate" else "🔴 Advanced"
    st.markdown(badge)
    st.markdown("**Speak Your Answer:**")

    if st.session_state.answers[q_index]:
        st.markdown("**Your Current Answer:**")
        st.success(st.session_state.answers[q_index])

    audio_data = mic_recorder(
        start_prompt="🎤 Start Recording Answer",
        stop_prompt="⏹ Stop Recording",
        just_once=True,
        key=f'recording_{q_index}',
        use_container_width=True
    )

    if audio_data:
        temp_audio_path = f"temp_input_{q_index}.wav"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_data['bytes'])
        if os.path.exists(temp_audio_path):
            model = whisper.load_model("base")
            result = model.transcribe(temp_audio_path)
            transcribed_text = result['text']
            st.session_state.answers[q_index] = transcribed_text
            os.remove(temp_audio_path)
            st.rerun()

    col1, col2 = st.columns(2)
    if q_index > 0 and col2.button("Previous Question"):
        st.session_state.current_q -= 1
        st.session_state.audio_played = False
        st.rerun()

    if col1.button("Next Question"):
        if q_index < len(st.session_state.questions) - 1:
            st.session_state.current_q += 1
            st.session_state.audio_played = False
            st.rerun()

    if st.button("🔊 Replay Question"):
        question_text = f"Question {q_index + 1}. {st.session_state.questions[q_index]}"
        audio_bytes = text_to_speech(question_text)
        autoplay_audio(audio_bytes)

    if q_index == len(st.session_state.questions) - 1:
        if st.button("✅ Submit Test"):
            with st.spinner("Evaluating answers..."):
                evaluations = []
                progress_bar = st.progress(0)
                for i, (q, a) in enumerate(zip(st.session_state.questions, st.session_state.answers)):
                    evaluations.append(strict_evaluation(q, a))
                    progress_bar.progress((i + 1) / len(st.session_state.questions))
                st.session_state.evaluations = evaluations
                st.session_state.submitted = True
                
                # Generate final feedback
                st.session_state.gemini_feedback = generate_final_feedback(
                    evaluations,
                    st.session_state.questions,
                    st.session_state.answers
                )
                
                st.rerun()

# ------------------------- RESULTS ---------------------
if st.session_state.submitted:
    st.header("📊 Detailed Evaluation Report")
    total_score = sum(eval['score'] for eval in st.session_state.evaluations)
    max_score = len(st.session_state.questions) * 5

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Score", f"{total_score}/{max_score}")
        col2.metric("Percentage", f"{(total_score/max_score)*100:.1f}%")
        col3.metric("Performance Level",
                    "Excellent" if total_score/max_score >= 0.8 else
                    "Good" if total_score/max_score >= 0.6 else
                    "Needs Practice")

    # Display detailed results in tabs
    if st.session_state.evaluations:
        tabs = st.tabs([f"Q{i+1}" for i in range(len(st.session_state.evaluations))])
        for i, tab in enumerate(tabs):
            with tab:
                eval_data = st.session_state.evaluations[i]
                st.markdown(f"**Question {i+1}:** {st.session_state.questions[i]}")
                st.code(st.session_state.answers[i] if st.session_state.answers[i].strip() else "(No answer given)")
                st.markdown(f"**Score:** {eval_data['score']}/5")
                st.progress(eval_data['score'] / 5)
                st.markdown("**Details:**")
                st.write(eval_data['justification'])
                st.markdown("**Reference Answer:**")
                st.info(eval_data['reference_answer'])

    # Show Gemini feedback
    if st.session_state.gemini_feedback:
        st.markdown("## 🧠 Gemini Feedback & Suggestions")
        st.write(st.session_state.gemini_feedback)

    if st.button("🔁 Retake Quiz"):
        st.session_state.questions = []
        st.session_state.answers = []
        st.session_state.evaluations = []
        st.session_state.submitted = False
        st.session_state.current_q = 0
        st.session_state.audio_played = False
        st.session_state.results_audio_generated = False
        st.session_state.gemini_feedback = ""
        st.rerun()

    if not st.session_state.get('results_audio_generated'):
        with st.spinner("Generating audio summary..."):
            summary = f"Your scored {total_score} out of {max_score}. "
            summary += "Excellent work!" if total_score/max_score >= 0.8 else \
                       "Good job!" if total_score/max_score >= 0.6 else \
                       "Keep practicing!"
            audio_bytes = text_to_speech(summary)
            autoplay_audio(audio_bytes)
            st.session_state.results_audio_generated = True