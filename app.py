#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
import streamlit as st
import requests
from datetime import datetime

# -------------------------
# Model Loader
# -------------------------
MODEL_DIR = "checkpoints"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
MODEL_URL = "https://github.com/hashimaliii/Kokoro-Chatbot-Encoder-Decoder-GenAi-Project-Python/raw/main/checkpoints/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def download_model():
    """Download model checkpoint if not available locally."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    with st.spinner("Downloading model weights..."):
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            st.success("✅ Model downloaded successfully!")
        else:
            st.error("❌ Failed to download model. Please check the link.")


@st.cache_resource
def load_model():
    """Load PyTorch model (Transformer)."""
    if not os.path.exists(MODEL_PATH):
        download_model()

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = checkpoint.get("model", None)

    if model is None:
        from model import TransformerModel
        model = TransformerModel(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])

    model.to(DEVICE)
    model.eval()
    return model


def generate_reply(model, tokenizer, emotion, user_input, max_len=50):
    """Generate response using greedy decoding."""
    model.eval()
    with torch.no_grad():
        input_text = f"Emotion: {emotion} | Situation: none | Customer: {user_input}\nAgent:"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)

        for _ in range(max_len):
            outputs = model(input_ids)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

        response = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return response.split("Agent:")[-1].strip()


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Emotion-Aware Chatbot", page_icon="💬", layout="centered")

st.markdown("<h1 style='text-align:center;'>💬 Emotion-Aware Chatbot</h1>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "clear_input" not in st.session_state:
    st.session_state["clear_input"] = False

emotion = st.selectbox("Choose Emotion:", ["happy", "sad", "angry", "neutral"])

# 🧩 Use key for text input
user_input = st.text_input(
    "Type your message:",
    key="user_input",
    value="" if st.session_state.clear_input else "",
)

# Reset flag right after rendering text_input
if st.session_state.clear_input:
    st.session_state.clear_input = False

# Send button logic
if st.button("Send") and user_input.strip():
    st.session_state["messages"].append({
        "role": "user",
        "content": user_input,
        "time": datetime.now().strftime("%H:%M:%S")
    })

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = load_model()
        response = generate_reply(model, tokenizer, emotion, user_input)
    except Exception as e:
        response = f"⚠️ Model error: {e}"

    st.session_state["messages"].append({
        "role": "bot",
        "content": response,
        "time": datetime.now().strftime("%H:%M:%S")
    })

    # ✅ Instead of directly clearing the input (which causes the crash),
    # set a flag and rerun
    st.session_state.clear_input = True
    st.rerun()


# -------------------------
# Chat History Display
# -------------------------
st.markdown("---")
for msg in st.session_state["messages"]:
    role_icon = "🙂" if msg["role"] == "user" else "🤖"
    bubble_color = "#1E90FF" if msg["role"] == "user" else "#333333"
    text_color = "#FFFFFF"
    st.markdown(
        f"""
        <div style='background-color:{bubble_color};
                    color:{text_color};
                    padding:10px;
                    border-radius:10px;
                    margin:5px 0;'>
            <b>{role_icon}</b> {msg['content']}<br>
            <small><i>{msg['time']}</i></small>
        </div>
        """,
        unsafe_allow_html=True,
    )
