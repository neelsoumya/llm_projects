import streamlit as st
import time
import random
from datetime import datetime
from collections import Counter
import re
# pip install -r requirements_chatbot_huggingface.txt

# deployed at
# https://huggingface.co/spaces/neelsoumya/opensourcechatbot

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NanoChat · LLM Playground",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark industrial background */
.stApp {
    background: #0d0d0f;
    color: #e8e4dc;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111114 !important;
    border-right: 1px solid #2a2a30;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em;
}

/* Chat messages */
.chat-msg {
    padding: 14px 18px;
    border-radius: 4px;
    margin: 8px 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    line-height: 1.7;
    border-left: 3px solid transparent;
}
.chat-msg.user {
    background: #1a1a1f;
    border-left-color: #f0c040;
    color: #e8e4dc;
}
.chat-msg.assistant {
    background: #141418;
    border-left-color: #4af0a0;
    color: #c8f0dc;
}
.chat-msg .role-label {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    opacity: 0.5;
    margin-bottom: 6px;
}

/* Metric cards */
.metric-card {
    background: #111114;
    border: 1px solid #2a2a30;
    border-radius: 4px;
    padding: 16px 20px;
    margin: 6px 0;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #f0c040;
    line-height: 1;
}
.metric-label {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: 0.45;
    margin-top: 4px;
}

/* Input box override */
.stTextInput > div > div > input, .stTextArea textarea {
    background: #111114 !important;
    color: #e8e4dc !important;
    border: 1px solid #2a2a30 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
}
.stTextInput > div > div > input:focus, .stTextArea textarea:focus {
    border-color: #f0c040 !important;
    box-shadow: 0 0 0 2px rgba(240,192,64,0.15) !important;
}

/* Buttons */
.stButton > button {
    background: #f0c040 !important;
    color: #0d0d0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 10px 24px !important;
}
.stButton > button:hover {
    background: #ffd760 !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #111114 !important;
    border: 1px solid #2a2a30 !important;
    color: #e8e4dc !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #2a2a30;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #888 !important;
    background: transparent !important;
    border: none !important;
    padding: 10px 24px !important;
}
.stTabs [aria-selected="true"] {
    color: #f0c040 !important;
    border-bottom: 2px solid #f0c040 !important;
}

/* Divider */
hr {
    border-color: #2a2a30 !important;
}

/* Spinner text */
.stSpinner > div {
    color: #4af0a0 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0d0f; }
::-webkit-scrollbar-thumb { background: #2a2a30; border-radius: 2px; }

/* Word freq bars */
.word-bar-container { margin: 4px 0; }
.word-bar-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #888;
    display: flex;
    justify-content: space-between;
    margin-bottom: 2px;
}
.word-bar {
    height: 6px;
    background: linear-gradient(90deg, #4af0a0, #f0c040);
    border-radius: 1px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "response_times" not in st.session_state:
    st.session_state.response_times = []
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0

# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_id: str):
    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        trust_remote_code=True,
    )
    return pipe

# ── Helpers ────────────────────────────────────────────────────────────────────
MODEL_OPTIONS = {
    "SmolLM2-135M-Instruct (HF)": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "SmolLM2-360M-Instruct (HF)": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "TinyLlama-1.1B-Chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
}

def count_tokens_approx(text: str) -> int:
    return max(1, len(text.split()) * 4 // 3)

def get_word_freq(messages, top_n=10):
    all_text = " ".join(m["content"] for m in messages).lower()
    words = re.findall(r"\b[a-z]{4,}\b", all_text)
    stopwords = {"that","this","with","from","have","will","been","they",
                 "what","when","your","just","more","also","some","than",
                 "then","there","their","these","those","about","which","would"}
    words = [w for w in words if w not in stopwords]
    return Counter(words).most_common(top_n)

def format_chat_history(messages, model_id: str):
    """Build a prompt string compatible with most instruct models."""
    if "SmolLM2" in model_id or "Qwen" in model_id:
        # ChatML format
        prompt = ""
        for m in messages:
            role = m["role"]
            content = m["content"]
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
    else:
        # TinyLlama / Llama-2 chat format
        prompt = "<s>"
        for m in messages:
            if m["role"] == "user":
                prompt += f"[INST] {m['content']} [/INST]"
            else:
                prompt += f" {m['content']} </s><s>"
    return prompt

def generate_response(pipe, messages, model_id, max_new_tokens, temperature):
    prompt = format_chat_history(messages, model_id)
    t0 = time.time()
    out = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=pipe.tokenizer.eos_token_id,
        return_full_text=False,
    )
    elapsed = time.time() - t0
    text = out[0]["generated_text"].strip()
    # Strip any trailing special tokens
    for tok in ["<|im_end|>", "</s>", "[INST]"]:
        text = text.split(tok)[0].strip()
    return text, elapsed

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ NanoChat")
    st.markdown("<p style='font-size:0.75rem;color:#666;font-family:Space Mono,monospace;margin-top:-8px'>Open-weight LLM Playground</p>", unsafe_allow_html=True)
    st.divider()

    selected_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
    model_id = MODEL_OPTIONS[selected_label]

    max_new_tokens = st.slider("Max new tokens", 32, 512, 200, 16)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)

    st.divider()

    if st.button("⚡  Load / Reload Model"):
        with st.spinner(f"Loading {selected_label}…"):
            try:
                st.session_state.pipeline = load_model(model_id)
                st.session_state.model_loaded = True
                st.success("Model ready!")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("🗑  Clear Chat"):
        st.session_state.messages = []
        st.session_state.total_tokens = 0
        st.session_state.response_times = []
        st.session_state.turn_count = 0
        st.rerun()

    st.divider()
    st.markdown(f"""
    <div style='font-family:Space Mono,monospace;font-size:0.68rem;color:#555;line-height:2'>
    Model ID<br>
    <span style='color:#f0c040'>{model_id.split("/")[-1]}</span><br><br>
    Status<br>
    <span style='color:{"#4af0a0" if st.session_state.model_loaded else "#f06060"}'>
    {"● Loaded" if st.session_state.model_loaded else "○ Not loaded"}
    </span>
    </div>
    """, unsafe_allow_html=True)

# ── Main area ──────────────────────────────────────────────────────────────────
tab_chat, tab_viz = st.tabs(["💬  Chat", "📊  Analytics"])

# ─── Chat tab ──────────────────────────────────────────────────────────────────
with tab_chat:
    st.markdown("<h1 style='margin-bottom:2px'>Chat</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.78rem;color:#555;font-family:Space Mono,monospace;margin-bottom:24px'>{model_id}</p>", unsafe_allow_html=True)

    if not st.session_state.model_loaded:
        st.info("👈  Load a model from the sidebar to begin.")
    else:
        # Render history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                role_label = "YOU" if msg["role"] == "user" else "AI"
                css_class = "user" if msg["role"] == "user" else "assistant"
                st.markdown(f"""
                <div class='chat-msg {css_class}'>
                  <div class='role-label'>{role_label}</div>
                  {msg['content']}
                </div>
                """, unsafe_allow_html=True)

        # Input
        with st.form("chat_form", clear_on_submit=True):
            cols = st.columns([8, 1])
            with cols[0]:
                user_input = st.text_area("Message", height=80, label_visibility="collapsed",
                                          placeholder="Type a message and press Send…")
            with cols[1]:
                submitted = st.form_submit_button("Send", use_container_width=True)

        if submitted and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input.strip()})
            st.session_state.total_tokens += count_tokens_approx(user_input)

            with st.spinner("Thinking…"):
                try:
                    reply, elapsed = generate_response(
                        st.session_state.pipeline,
                        st.session_state.messages,
                        model_id,
                        max_new_tokens,
                        temperature,
                    )
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.session_state.total_tokens += count_tokens_approx(reply)
                    st.session_state.response_times.append(round(elapsed, 2))
                    st.session_state.turn_count += 1
                except Exception as e:
                    st.error(f"Generation error: {e}")
            st.rerun()

# ─── Analytics tab ─────────────────────────────────────────────────────────────
with tab_viz:
    st.markdown("<h1 style='margin-bottom:2px'>Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.78rem;color:#555;font-family:Space Mono,monospace;margin-bottom:24px'>Session insights</p>", unsafe_allow_html=True)

    msgs = st.session_state.messages
    rt   = st.session_state.response_times

    # ── Metrics row ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-value'>{st.session_state.turn_count}</div>
          <div class='metric-label'>Turns</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-value'>{st.session_state.total_tokens}</div>
          <div class='metric-label'>Est. Tokens</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        avg_rt = round(sum(rt)/len(rt), 2) if rt else 0.0
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-value'>{avg_rt}s</div>
          <div class='metric-label'>Avg Response</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        user_msgs = [m for m in msgs if m["role"]=="user"]
        avg_len = round(sum(len(m["content"].split()) for m in user_msgs)/len(user_msgs)) if user_msgs else 0
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-value'>{avg_len}</div>
          <div class='metric-label'>Avg User Words</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    col_left, col_right = st.columns(2)

    # ── Response time chart ────────────────────────────────────────────────────
    with col_left:
        st.markdown("#### Response times (s)")
        if rt:
            import pandas as pd
            df_rt = pd.DataFrame({"Turn": list(range(1, len(rt)+1)), "Seconds": rt})
            st.line_chart(df_rt.set_index("Turn"), color="#4af0a0", height=200)
        else:
            st.caption("No data yet — start chatting!")

    # ── Message length chart ───────────────────────────────────────────────────
    with col_right:
        st.markdown("#### Message lengths (words)")
        if msgs:
            import pandas as pd
            rows = []
            u_idx = a_idx = 1
            for m in msgs:
                wc = len(m["content"].split())
                if m["role"] == "user":
                    rows.append({"idx": u_idx, "role": "User", "words": wc})
                    u_idx += 1
                else:
                    rows.append({"idx": a_idx, "role": "AI", "words": wc})
                    a_idx += 1
            import pandas as pd
            df_ml = pd.DataFrame(rows)
            st.bar_chart(df_ml.pivot_table(index="idx", columns="role", values="words", aggfunc="sum").fillna(0),
                         color=["#f0c040", "#4af0a0"], height=200)
        else:
            st.caption("No data yet — start chatting!")

    st.divider()

    # ── Word frequency ──────────────────────────────────────────────────────────
    st.markdown("#### Top words across conversation")
    if msgs:
        freq = get_word_freq(msgs, top_n=12)
        if freq:
            max_count = freq[0][1]
            for word, count in freq:
                pct = int((count / max_count) * 100)
                st.markdown(f"""
                <div class='word-bar-container'>
                  <div class='word-bar-label'><span>{word}</span><span>{count}</span></div>
                  <div class='word-bar' style='width:{pct}%'></div>
                </div>""", unsafe_allow_html=True)
    else:
        st.caption("No data yet — start chatting!")

    st.divider()

    # ── Role distribution ────────────────────────────────────────────────────
    st.markdown("#### Message distribution")
    if msgs:
        u_count = sum(1 for m in msgs if m["role"]=="user")
        a_count = sum(1 for m in msgs if m["role"]=="assistant")
        total   = u_count + a_count
        u_pct   = int(u_count/total*100)
        a_pct   = 100 - u_pct
        st.markdown(f"""
        <div style='display:flex;gap:0;border-radius:3px;overflow:hidden;height:28px;margin:8px 0'>
          <div style='width:{u_pct}%;background:#f0c040;display:flex;align-items:center;
                      justify-content:center;font-family:Space Mono,monospace;
                      font-size:0.7rem;color:#0d0d0f;font-weight:700'>
            USER {u_pct}%
          </div>
          <div style='width:{a_pct}%;background:#4af0a0;display:flex;align-items:center;
                      justify-content:center;font-family:Space Mono,monospace;
                      font-size:0.7rem;color:#0d0d0f;font-weight:700'>
            AI {a_pct}%
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.caption("No data yet.")
