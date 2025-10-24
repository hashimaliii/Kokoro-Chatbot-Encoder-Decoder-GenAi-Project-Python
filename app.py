import os
import time
import csv
import torch
import streamlit as st
import sentencepiece as spm
import requests

from model import TransformerSeq2Seq, VOCAB_SIZE, DEVICE, MAX_TARGET_LEN, generate_greedy, generate_beam

# --------------------------------
# Fix Streamlit file watcher issue
# --------------------------------
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# --------------------------------
# GitHub Raw URLs (for auto-download)
# --------------------------------
BASE_URL = "https://raw.githubusercontent.com/hashimaliii/Kokoro-Chatbot-Encoder-Decoder-GenAi-Project-Python/main"
CHECKPOINT_DIR = "checkpoints"
SP_MODEL = os.path.join("preprocessed", "spm_emotion.model")
BEST_MODEL = os.path.join(CHECKPOINT_DIR, "best_model.pth")

SP_MODEL_URL = f"{BASE_URL}/preprocessed/spm_emotion.model"
BEST_MODEL_URL = f"{BASE_URL}/checkpoints/best_model.pth"

# --------------------------------
# Utility: download file if not found
# --------------------------------
def ensure_file(local_path, url):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        st.warning(f"Downloading {os.path.basename(local_path)} from GitHub...")
        r = requests.get(url)
        if r.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(r.content)
            st.success(f"✅ Downloaded {os.path.basename(local_path)}")
        else:
            st.error(f"Failed to download {os.path.basename(local_path)} ({r.status_code})")

# Ensure model files exist
ensure_file(SP_MODEL, SP_MODEL_URL)
ensure_file(BEST_MODEL, BEST_MODEL_URL)

# --------------------------------
# Streamlit App Config
# --------------------------------
st.set_page_config(page_title="Emotion-Aware Chatbot", layout="centered")
st.title("💬 Emotion-Aware Chatbot")

# --------------------------------
# Cached resources
# --------------------------------
@st.cache_resource
def load_tokenizer(path):
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp

@st.cache_resource
def load_model(path):
    model = TransformerSeq2Seq(VOCAB_SIZE).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

# Load resources
sp = load_tokenizer(SP_MODEL)
model = load_model(BEST_MODEL)

# --------------------------------
# Sidebar
# --------------------------------
st.sidebar.header("⚙️ Settings")
decoding = st.sidebar.radio("Decoding method", ["Greedy", "Beam"])
beam_size = st.sidebar.slider("Beam size", 2, 10, 4)

checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]
if checkpoint_files:
    selected_ckpt = st.sidebar.selectbox("Available checkpoints", checkpoint_files)
    if st.sidebar.button("Load selected checkpoint"):
        ensure_file(os.path.join(CHECKPOINT_DIR, selected_ckpt),
                    f"{BASE_URL}/checkpoints/{selected_ckpt}")
        model = load_model(os.path.join(CHECKPOINT_DIR, selected_ckpt))
        st.sidebar.success(f"Loaded: {selected_ckpt}")

if st.sidebar.button("🗑️ Clear conversation"):
    st.session_state.clear()
    st.rerun()

if st.sidebar.button("💾 Save conversation"):
    conv = st.session_state.get("history", [])
    if conv:
        fname = f"conversation_{int(time.time())}.csv"
        with open(fname, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["role", "text"])
            writer.writeheader()
            writer.writerows(conv)
        st.sidebar.success(f"Saved {fname}")

# --------------------------------
# Helpers
# --------------------------------
def encode_input(sp_path, text):
    sp_proc = spm.SentencePieceProcessor()
    sp_proc.load(sp_path)
    return sp_proc.encode(text, out_type=int)

def decode_output(ids):
    return sp.decode([i for i in ids if i not in [sp.pad_id(), sp.bos_id(), sp.eos_id()]])

# --------------------------------
# Session state setup
# --------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "emotion" not in st.session_state:
    st.session_state.emotion = "neutral"

# --------------------------------
# Emotion context
# --------------------------------
with st.sidebar.expander("🧠 Emotion Context"):
    st.session_state.emotion = st.selectbox(
        "Select Emotion",
        ["neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted"],
        index=["neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted"].index(st.session_state.emotion)
    )

# --------------------------------
# Chat display
# --------------------------------
for msg in st.session_state.history:
    if msg["role"] == "customer":
        with st.chat_message("user", avatar="🙂"):
            st.markdown(msg["text"])
            st.caption(msg.get("time", ""))
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["text"])
            st.caption(msg.get("time", ""))

# --------------------------------
# Input box (persistent bottom)
# --------------------------------
prompt = st.chat_input("Type your message...")

if prompt:
    current_time = time.strftime("%H:%M:%S")
    st.session_state.history.append({"role": "customer", "text": prompt, "time": current_time})

    with st.chat_message("user", avatar="🙂"):
        st.markdown(prompt)
        st.caption(current_time)

    emotion = st.session_state.emotion
    model_input = f"Emotion: {emotion} | Customer: {prompt}\nAgent:"

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            src_ids = encode_input(SP_MODEL, model_input)
            src_tensor = torch.tensor(src_ids, dtype=torch.long).to(DEVICE)

            if decoding.lower() == "greedy":
                pred_ids = generate_greedy(model, src_tensor, max_len=MAX_TARGET_LEN)
            else:
                pred_ids = generate_beam(model, src_tensor, beam_size=beam_size, max_len=MAX_TARGET_LEN)

            reply = decode_output(pred_ids)
            reply_time = time.strftime("%H:%M:%S")

            st.markdown(reply)
            st.caption(reply_time)

    st.session_state.history.append({"role": "agent", "text": reply, "time": reply_time})
