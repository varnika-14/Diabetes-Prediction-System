import streamlit as st
import requests
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(os.path.join('..', 'backend', '.env'))

def get_mistral_response(user_message):
    api_key = os.getenv("MISTRAL_API_KEY")
    
    if not api_key:
        return "Mistral AI service is not configured. Please contact the administrator."
    
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json"
    }
    
    system_prompt = """You are a compassionate, knowledgeable diabetes assistant. 
    Provide accurate, evidence-based information about diabetes in simple terms.
    Always remind users to consult healthcare professionals for personalized advice.
    Keep responses helpful, supportive, and concise (under 150 words)."""
    
    data = {
        "model": "mistral-tiny",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.RequestException:
        return "I'm having trouble connecting to my AI service. Please try again later."
    except Exception:
        return "Sorry, I encountered an error. Please try again."

def show_chatbot():
    st.markdown("## Chat with Diabetes Assistant")
    st.markdown("Ask me anything about diabetes, prevention, diet, or lifestyle.")
    
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    for msg in st.session_state.chat_messages:
        if msg['role'] == 'user':
            st.chat_message("user").write(msg['content'])
        else:
            st.chat_message("assistant").write(msg['content'])
    
    prompt = st.chat_input("Ask a question about diabetes...")
    if prompt:
        st.session_state.chat_messages.append({'role': 'user', 'content': prompt})
        with st.spinner("Thinking..."):
            response = get_mistral_response(prompt)
            st.session_state.chat_messages.append({'role': 'assistant', 'content': response})
        st.rerun()
    
    st.markdown("### Quick Questions")
    quick_questions = [
        "What is diabetes?",
        "How to prevent diabetes?",
        "What are normal glucose levels?",
        "How often should I get screened?",
        "What are the symptoms of diabetes?",
        "What lifestyle changes reduce risk?",
        "What is a healthy diet for diabetes?",
        "How does exercise help diabetes?"
    ]
    
    cols = st.columns(4)
    for i, q in enumerate(quick_questions[:8]):
        col_idx = i % 4
        if cols[col_idx].button(q, key="quick_" + str(i)):
            st.session_state.chat_messages.append({'role': 'user', 'content': q})
            with st.spinner("Thinking..."):
                response = get_mistral_response(q)
                st.session_state.chat_messages.append({'role': 'assistant', 'content': response})
            st.rerun()
    
    if st.button("Clear Chat History"):
        st.session_state.chat_messages = []
        st.rerun()