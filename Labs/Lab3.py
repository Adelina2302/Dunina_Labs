import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError

# try to import tiktoken; fallback if not available
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except Exception:
    TIKTOKEN_AVAILABLE = False

# ------------------------
# Setup
# ------------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Lab 3 — Token Buffer Chatbot", layout="wide")
st.title("Lab 3: Streaming Chatbot")

# ------------------------
# Helpers
# ------------------------
def style_instruction(style: str) -> str:
    if style == "100 words":
        return "Answer in about 100 words."
    if style == "2 connected paragraphs":
        return "Answer in exactly two connected paragraphs."
    if style == "5 bullet points":
        return "Provide exactly five concise bullet points."
    return ""

def safe_encode_len(text: str, model: str = "gpt-4o-mini"):
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    else:
        return max(1, len(text) // 4)  # rough estimate

def count_tokens(messages, model="gpt-4o-mini"):
    tokens = 0
    for msg in messages:
        tokens += 4
        tokens += safe_encode_len(msg.get("content", ""), model)
    return tokens

def trim_messages_to_token_limit(messages, max_tokens, model="gpt-4o-mini"):
    trimmed = list(messages)
    total_tokens = count_tokens(trimmed, model)
    while trimmed and total_tokens > max_tokens:
        trimmed.pop(0)
        total_tokens = count_tokens(trimmed, model)
    return trimmed

def trim_messages_last_n_pairs(messages, n_pairs=2):
    if n_pairs <= 0:
        return []
    tail = []
    user_count = 0
    for m in reversed(messages):
        tail.append(m)
        if m.get("role") == "user":
            user_count += 1
            if user_count >= n_pairs:
                break
    return list(reversed(tail))

def is_affirmative(text: str) -> bool:
    s = text.strip().lower()
    affirmatives = {"yes","y","да","дa","si","sí","oui","ja","sure","yeah","yep","ok","okay","please","давай"}
    return any(token in s.split() for token in affirmatives)

def is_negative(text: str) -> bool:
    s = text.strip().lower()
    negatives = {"no","n","нет","non","nein","nope","nah"}
    return any(token in s.split() for token in negatives)

def generate_stream_response(client, model, messages_for_model):
    response_text = ""
    stream = client.chat.completions.create(
        model=model,
        messages=messages_for_model,
        stream=True,
        temperature=0.7,
    )
    try:
        return st.write_stream(stream)
    except Exception:
        placeholder = st.empty()
        for event in stream:
            chunk_text = ""
            try:
                chunk_text = event.choices[0].delta.content
            except Exception:
                try:
                    choices = getattr(event, "choices", None) or event.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        chunk_text = delta.get("content", "") or ""
                except Exception:
                    chunk_text = ""
            response_text += chunk_text
            placeholder.markdown(response_text)
        return response_text

# ------------------------

# ------------------------
if not openai_api_key:
    st.error("OPENAI_API_KEY not found in environment variables or .env file.")
else:
    try:
        client = OpenAI(api_key=openai_api_key)
    except AuthenticationError:
        st.error("Invalid API key.")
        client = None

    if client:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "awaiting_more_info" not in st.session_state:
            st.session_state.awaiting_more_info = False

        # Sidebar controls
        st.sidebar.header("Chatbot Controls")
        language = st.sidebar.selectbox(
            "Output language",
            ["English", "Русский", "Español", "Français", "Deutsch", "中文", "日本語"],
            index=0,
            key="chat_lang",
        )
        style = st.sidebar.selectbox(
            "Response style",
            ["Normal", "100 words", "2 connected paragraphs", "5 bullet points"],
            index=0,
            key="chat_style",
        )

        use_advanced = st.sidebar.checkbox("Use Advanced Model (gpt-4o)", value=False)
        model = "gpt-4o" if use_advanced else "gpt-4o-mini"
        st.sidebar.markdown(f"**Selected model:** `{model}`")

        buffer_mode = st.sidebar.selectbox("Buffer mode", ["token_limit (recommended)", "last_n_pairs"], index=0)
        if buffer_mode.startswith("token"):
            max_tokens = st.sidebar.slider("Max tokens in buffer", 200, 3000, 800, 100)
        else:
            last_n = st.sidebar.slider("Keep last N user messages (pairs)", 1, 10, 2, 1)

        if st.sidebar.button("Clear chat history", key="clear_history"):
            st.session_state.messages = []
            st.session_state.awaiting_more_info = False

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Main user input
        if prompt := st.chat_input("Ask me anything..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            base_system = (
                f"Write the output in {language}. "
                "Use simple language suitable for a 10-year-old: short sentences, clear examples. "
                f"{style_instruction(style)}"
            )

            if st.session_state.awaiting_more_info:
                if is_affirmative(prompt):
                    more_info_prompt = (
                        "The user said YES and wants more information. "
                        "Expand your previous answer with more detail, examples, and simple explanations for a 10-year-old. "
                        "After the explanation, ask again: 'DO YOU WANT MORE INFO? (yes/no)'."
                    )
                    trimmed = (
                        trim_messages_to_token_limit(st.session_state.messages, max_tokens, model)
                        if buffer_mode.startswith("token")
                        else trim_messages_last_n_pairs(st.session_state.messages, last_n)
                    )
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            try:
                                req_msgs = [{"role": "system", "content": base_system}] + trimmed + [{"role": "user", "content": more_info_prompt}]
                                response_text = generate_stream_response(client, model, req_msgs)
                            except Exception as e:
                                st.error(f"Error: {e}")
                                response_text = ""
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        st.session_state.awaiting_more_info = True
                elif is_negative(prompt):
                    assistant_text = "Okay! What question can I help you with next?"
                    with st.chat_message("assistant"):
                        st.markdown(assistant_text)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
                    st.session_state.awaiting_more_info = False
                    st.stop()  # stop further processing here
                else:
                    st.session_state.awaiting_more_info = False

            if not st.session_state.awaiting_more_info:
                system_instruction = base_system
                trimmed_messages = (
                    trim_messages_to_token_limit(st.session_state.messages, max_tokens, model)
                    if buffer_mode.startswith("token")
                    else trim_messages_last_n_pairs(st.session_state.messages, last_n)
                )
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            req_msgs = [{"role": "system", "content": system_instruction}] + trimmed_messages
                            response_text = generate_stream_response(client, model, req_msgs)
                        except Exception as e:
                            st.error(f"Error: {e}")
                            response_text = ""
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                    follow_up = "DO YOU WANT MORE INFO? (yes/no)"
                    if "рус" in language.lower():
                        follow_up = "Хотите больше информации? (да/нет)"
                    with st.chat_message("assistant"):
                        st.markdown(follow_up)
                    st.session_state.messages.append({"role": "assistant", "content": follow_up})
                    st.session_state.awaiting_more_info = True

       