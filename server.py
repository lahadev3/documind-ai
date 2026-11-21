
import streamlit as st
import os
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

st.set_page_config(page_title="DocuMind AI", page_icon="🤖")

st.title("🤖 NVIDIA RAG Chatbot")
st.caption("Upload documents and ask questions about them.")

# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# ---------------- API KEY ----------------

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    st.error("Please add NVIDIA_API_KEY in your .env file")
    st.stop()

# ---------------- CHAT MODEL ----------------

llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    api_key=api_key,
    temperature=0.2,
    max_tokens=1024
)

# ---------------- LOCAL EMBEDDING MODEL ----------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# ---------------- CREATE VECTOR DATABASE ----------------

def create_vector_db(chunks):

    try:
        vector_db = FAISS.from_documents(
            chunks,
            embeddings
        )

        return vector_db

    except Exception as e:
        st.error(f"Vector database error: {e}")
        return None


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("📁 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT or DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    # PROCESS DOCUMENTS

    if st.button("📚 Process Documents", use_container_width=True):

        if not uploaded_files:

            st.warning("Upload some documents first.")

        else:

            documents = []

            try:

                with st.spinner("Reading documents..."):

                    for file in uploaded_files:

                        file_path = f"temp_{file.name}"

                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())

                        if file.name.lower().endswith(".pdf"):
                            loader = PyPDFLoader(file_path)

                        elif file.name.lower().endswith(".txt"):
                            loader = TextLoader(
                                file_path,
                                encoding="utf-8"
                            )

                        elif file.name.lower().endswith(".docx"):
                            loader = Docx2txtLoader(file_path)

                        documents.extend(loader.load())

                        os.remove(file_path)

                # SPLIT DOCUMENTS

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )

                chunks = splitter.split_documents(documents)

                if not chunks:

                    st.error("No readable text found in documents.")

                else:

                    st.info(f"Total chunks: {len(chunks)}")

                    # CREATE FAISS DATABASE

                    with st.spinner("Creating local embeddings..."):

                        vector_db = create_vector_db(chunks)

                    if vector_db is not None:

                        st.session_state.vector_db = vector_db

                        st.session_state.processed_files = [
                            file.name for file in uploaded_files
                        ]

                        st.session_state.messages = []

                        st.success("Documents processed successfully!")

            except Exception as e:

                st.error(f"Document processing error: {e}")

    # CLEAR CHAT

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()

    # CLEAR KNOWLEDGE BASE

    if st.button("🧹 Clear Knowledge Base", use_container_width=True):

        st.session_state.vector_db = None
        st.session_state.messages = []
        st.session_state.processed_files = []

        st.rerun()


# ---------------- DOCUMENT STATUS ----------------

if st.session_state.vector_db is not None:

    st.success("📄 Knowledge Base Ready")

    with st.expander("View processed documents"):

        for name in st.session_state.processed_files:
            st.write(f"📄 {name}")


# ---------------- CHAT HISTORY ----------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------- USER INPUT ----------------

question = st.chat_input("Ask something about your documents...")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # CHECK KNOWLEDGE BASE

    if st.session_state.vector_db is None:

        answer = "Please upload and process your documents first."

        with st.chat_message("assistant"):
            st.warning(answer)

    else:

        try:

            with st.chat_message("assistant"):

                with st.spinner("Searching documents..."):

                    # RETRIEVE DOCUMENTS

                    docs = st.session_state.vector_db.similarity_search(
                        question,
                        k=3
                    )

                    context = "\n\n".join(
                        doc.page_content for doc in docs
                    )

                    # PROMPT

                    prompt = f"""
You are a helpful RAG assistant.

Answer the question using the context below.
If the answer is not present in the context, say:
"I could not find this information in the uploaded documents."

Do not invent information.

Context:
{context}

Question:
{question}
"""

                    # GENERATE ANSWER

                    response = llm.invoke([
                        SystemMessage(
                            content="Answer clearly and accurately using the provided context."
                        ),
                        HumanMessage(content=prompt)
                    ])

                    answer = response.content

                    st.markdown(answer)

        except Exception as e:

            answer = f"Error generating response: {e}"

            with st.chat_message("assistant"):
                st.error(answer)

    # SAVE RESPONSE

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


# ---------------- KNOWLEDGE BASE ----------------

if st.session_state.vector_db is not None:

    with st.expander("🔍 Knowledge Base Information"):

        st.write("Vector database: FAISS")
        st.write("Embedding model: all-MiniLM-L6-v2")
        st.write("Embedding mode: Local CPU")
        st.write("Chat model: NVIDIA Nemotron")

