# I chose the multimodal chatbot lab because 
# I was interested in seeing how this works in the context 
# of a Streamlit interface and whether we could use 
# a similar approach in our AI Tax Assistant project.
# For example, users could upload an image of a tax form 
# when they’re unsure what it is or want to understand it 
# better, and the chatbot could identify the form and 
# explain its purpose. 
# The same approach could also be used for audio - 
# users could ask questions by voice instead of typing.

import streamlit as st
import openai
import requests

st.title("Lab 8 – Multimodal Chatbot (Images & Audio)")

openai.api_key = (
    st.secrets["OPENAI_API_KEY"]
    if "OPENAI_API_KEY" in st.secrets
    else st.text_input("Enter OPENAI_API_KEY", type="password")
)

def analyze_image_with_text(prompt, image_url):
    """Send prompt and image to OpenAI multimodal model."""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }]
    )
    return response.choices[0].message.content

def analyze_audio_with_text(prompt, audio_url):
    """Step 1: Transcribe, Step 2: Reason with GPT"""
    temp_audio = "temp_audio.mp3"
    with open(temp_audio, "wb") as file:
        file.write(requests.get(audio_url).content)
    with open(temp_audio, "rb") as afile:
        transcript = openai.audio.transcriptions.create(model="whisper-1", file=afile)
    st.markdown("**Transcript:**")
    st.code(transcript.text)
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"{prompt}\n\nAudio Transcript: {transcript.text}"
        }]
    )
    return resp.choices[0].message.content

with st.form("multimodal_form"):
    prompt = st.text_input("Your question or instruction")
    image_url = st.text_input("Image URL (e.g., https://www.placecats.com/300/200)")
    audio_url = st.text_input("Audio URL (mp3, optional)")
    submitted = st.form_submit_button("Analyze")

    if image_url:
        st.image(image_url, caption="Image from URL", use_container_width=True)

if submitted:
    if image_url and prompt:
        with st.spinner("Analyzing image..."):
            try:
                result = analyze_image_with_text(prompt, image_url)
                st.success(result)
            except Exception as e:
                st.error(f"Error analyzing image: {e}")
    elif audio_url and prompt:
        with st.spinner("Analyzing audio..."):
            try:
                result = analyze_audio_with_text(prompt, audio_url)
                st.success(result)
            except Exception as e:
                st.error(f"Error analyzing audio: {e}")
    else:
        st.info("Please provide at least an image or audio URL and a prompt.")

with st.expander("Discussion / Reflections"):
    st.write("""
This chatbot can 'see' and 'hear'. Using OpenAI's newest models, you can ask about images (what's happening, what objects are present, is the meal healthy, etc.) or even analyze/summarize audio.
- For images: The model mixes your question and the visual info for answers.
- For audio: Whisper transcribes, then GPT summarizes/reasons.
- Try with [random cats](https://www.placecats.com/300/200) or sounds from [pdsounds.tuxfamily.org](https://pdsounds.tuxfamily.org)
""")
