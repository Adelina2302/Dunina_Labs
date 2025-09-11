import streamlit as st
import os
import time
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError
from io import BytesIO
from typing import List
from pypdf import PdfReader

load_dotenv()

def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    texts: List[str] = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n\n".join(texts).strip()

def style_instruction(style: str) -> str:
    if style == "100 words":
        return "Produce a single coherent summary of about 100 words."
    if style == "2 connected paragraphs":
        return "Produce a concise summary as exactly two connected paragraphs."
    if style == "5 bullet points":
        return "Produce exactly five concise bullet points."
    return "Produce a concise summary."

def build_prompt(language: str, style: str, text: str) -> str:
    return f"""
You are a careful academic summarizer.
Write the output in: {language}.
{style_instruction(style)}

SOURCE TEXT START
{text}
SOURCE TEXT END
""".strip()

def llm_summarize(client, model: str, language: str, style: str, text: str) -> str:
    prompt = build_prompt(language, style, text)
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.2,
    )
    return resp.output_text.strip()

# ---- Lab2 page ----
st.title("📄 PDF Summarizer (Lab 2c)")

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY not found in environment variables or .env file.")
else:
    try:
        client = OpenAI(api_key=openai_api_key)
    except AuthenticationError:
        st.error("Invalid API key.")
        client = None

    if client:
        st.sidebar.header("Lab 2c Controls")
        language = st.sidebar.selectbox(
            "Output language",
            ["English", "Русский", "Español", "Français", "Deutsch", "中文", "日本語"],
            index=0,
        )
        summary_style = st.sidebar.selectbox(
            "Summary style",
            ["100 words", "2 connected paragraphs", "5 bullet points"],
            index=0,
        )
        use_advanced = st.sidebar.checkbox("Use Advanced Model (gpt-4o)", value=False)
        model = "gpt-4o" if use_advanced else "gpt-4o-mini"
        st.sidebar.markdown(f"**Selected model:** `{model}`")

        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_file:
            with st.spinner("Extracting text..."):
                text = extract_pdf_text(uploaded_file.getvalue())
            if text:
                start = time.time()
                with st.spinner(f"Summarizing with {model}..."):
                    summary = llm_summarize(client, model, language, summary_style, text)
                elapsed = time.time() - start
                st.subheader("Summary")
                st.write(summary)
                st.metric("Time taken", f"{elapsed:.1f} s")
                st.download_button(
                    "Download summary as .txt",
                    data=summary.encode("utf-8"),
                    file_name=uploaded_file.name.replace(".pdf", "") + "_summary.txt",
                    mime="text/plain",
                )
            else:
                st.error("No extractable text found in PDF.")
