#I chose to complete the LangChain Lab 
#because it allowed me to explore a more advanced agent 
# architecture with real retrieval, memory, and reasoning capabilities. 
# I wanted hands-on experience building a full RAG system using LangChain tools,
# which felt closer to real-world AI research assistant applications.

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# 🧠 Lab6.py — LangChain Research Assistant (Vector DB from CSV)
import streamlit as st
import os
import glob
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
st.set_page_config(page_title="Lab 6 — Research Assistant", layout="wide")
st.title("🧠 Lab 6: LangChain Research Assistant — Vector DB from CSV")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Add it to your .env file.")
    st.stop()

# ------------------------------------------------------------
# CSV discovery
# ------------------------------------------------------------
def find_csv_file():
    patterns = ["arxiv_papers_extended*.csv", "*arxiv*.csv"]
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            return matches[0]
    for folder in ["./Labs", "./data"]:
        for p in patterns:
            matches = glob.glob(os.path.join(folder, p))
            if matches:
                return matches[0]
    return None

CSV_PATH = find_csv_file()
st.sidebar.markdown("**Detected CSV file:**")
st.sidebar.write(CSV_PATH or "Not found")

if CSV_PATH is None:
    st.error("CSV file not found. Place arxiv_papers_extended_*.csv in the root or Labs folder.")
    st.stop()

# ------------------------------------------------------------
# Initialize vectorstore (from CSV only)
# ------------------------------------------------------------
@st.cache_resource
def initialize_vectorstore_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    st.info("Building Chroma vectorstore from CSV (title + abstract)...")

    docs = []
    for _, row in df.iterrows():
        title = row.get("title", "") or ""
        authors = row.get("authors", "") or ""
        abstract = row.get("abstract", "") or ""
        year = row.get("year", "") or ""
        category = row.get("category", "") or ""
        venue = row.get("venue", "") or ""
        link = row.get("link", "") or row.get("pdf_url", "") or row.get("canonical_link", "")

        text = (
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Abstract: {abstract}\n"
            f"Year: {year}\n"
            f"Category: {category}\n"
            f"Venue: {venue}\n"
        )
        docs.append(Document(page_content=text, metadata={"title": title, "authors": authors, "link": link}))

    PERSIST_DIR = "LAB6_vector_db"
    os.makedirs(PERSIST_DIR, exist_ok=True)

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIR)
    st.success("Vectorstore built successfully.")
    return vectorstore, df

# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
st.sidebar.header("Vector DB Controls")
st.sidebar.markdown("**Index source:** From CSV (fast)")
rebuild_btn = st.sidebar.button("Build vector DB from CSV")

if "lab6_vectorstore" not in st.session_state or rebuild_btn:
    try:
        st.session_state.lab6_vectorstore, st.session_state.lab6_df = initialize_vectorstore_from_csv(CSV_PATH)
    except Exception as e:
        st.error(f"Vectorstore Error: {e}")
        st.stop()

# ------------------------------------------------------------
# Tools: search and compare
# ------------------------------------------------------------
def search_papers(query: str) -> str:
    try:
        results = st.session_state.lab6_vectorstore.similarity_search(query, k=5)
    except Exception as e:
        return f"Error during search: {e}"
    if not results:
        return f"No papers found about '{query}'"

    lines = []
    for i, doc in enumerate(results):
        md = doc.metadata or {}
        title = md.get("title", "No title")
        authors = md.get("authors", "")
        link = md.get("link", "")
        snippet = (doc.page_content[:600].replace("\n", " ")) if doc.page_content else ""
        lines.append(f"**{i+1}. {title}**\nAuthors: {authors}\nLink: {link}\n\nSnippet: {snippet}\n")
    return "\n\n".join(lines)

def compare_papers(query: str) -> str:
    import re
    parts = re.split(r"\s+and\s+|\s+vs\.?\s+", query)
    if len(parts) < 2:
        return "Please specify two paper titles, e.g.: 'Transformers and BERT'"

    df = st.session_state.lab6_df

    def find(title):
        m = df[df["title"].str.contains(title.strip(), case=False, na=False)]
        if m.empty:
            return None
        r = m.iloc[0].to_dict()
        link = r.get("link") or r.get("pdf_url") or r.get("canonical_link") or ""
        abstract = r.get("abstract", "")
        return f"**{r.get('title','')}**\nAuthors: {r.get('authors','')}\nYear: {r.get('year','')}\nLink: {link}\n\nAbstract: {abstract[:1500]}..."

    p1 = find(parts[0])
    p2 = find(parts[1])
    if not p1 or not p2:
        return "Could not find one or both papers by title. Try a different substring."
    return f"### Paper 1\n{p1}\n\n### Paper 2\n{p2}"

tools = [
    Tool(name="SearchPapers", func=search_papers, description="Find research papers on a topic"),
    Tool(name="ComparePapers", func=compare_papers, description="Compare two papers by title"),
]

# ------------------------------------------------------------
# Agent setup
# ------------------------------------------------------------
if "lab6_agent" not in st.session_state:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="output")
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    st.session_state.lab6_agent = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

# ------------------------------------------------------------
# Chat UI
# ------------------------------------------------------------
st.sidebar.header("Chat Controls")
if "lab6_messages" not in st.session_state:
    st.session_state.lab6_messages = []
if st.sidebar.button("Clear chat history"):
    st.session_state.lab6_messages = []
    st.experimental_rerun()

st.header("Chat with Research Assistant")
for msg in st.session_state.lab6_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about research papers or compare two papers...")
if user_input:
    st.session_state.lab6_messages.append({"role": "user", "content": user_input})
    with st.chat_message("assistant"):
        with st.spinner("Agent thinking..."):
            try:
                response = st.session_state.lab6_agent.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.lab6_messages
                })
                output = response.get("output", str(response))
            except Exception as e:
                output = f"Error during agent run: {e}"
        st.session_state.lab6_messages.append({"role": "assistant", "content": output})
        st.markdown(output)

st.markdown("---")
st.markdown(
    "**Usage tips:**\n"
    "- Try `find papers about reinforcement learning`\n"
    "- Compare with `compare BERT and GPT`\n"
    "- Place your CSV file in the project root and click *Build vector DB from CSV*"
)
