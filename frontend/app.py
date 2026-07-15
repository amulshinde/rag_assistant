import requests
import streamlit as st

from streaming_utils import iter_response_chunks

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

BACKEND_URL = "http://backend:8000"

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide"
)

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.title("📄 Document Center")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file:

        st.success(uploaded_file.name)

        # Upload only when a new document is selected
        if uploaded_file.name != st.session_state.uploaded_filename:

            with st.spinner("Generating embeddings..."):

                try:

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf"
                        )
                    }

                    response = requests.post(
                        f"{BACKEND_URL}/generate_embeddings",
                        files=files
                    )

                    if response.status_code == 200:

                        data = response.json()

                        st.session_state.pdf_name = data["filename"]

                        st.session_state.uploaded_filename = uploaded_file.name

                        # Clear previous conversation
                        st.session_state.messages = []

                        st.success("Embeddings generated successfully.")

                    else:

                        st.error(response.text)

                except Exception as e:

                    st.error(e)

    else:

        st.info("Upload a PDF to begin.")

# ---------------------------------------------------------
# MAIN PAGE
# ---------------------------------------------------------

st.title("🤖 PDF RAG Assistant")

st.caption("Ask questions about your uploaded document.")

# ---------------------------------------------------------
# Display Chat History
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the document..."):

    if st.session_state.pdf_name is None:
        st.error("Please upload a PDF first.")
        st.stop()

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user",  avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):

        try:
            response = requests.post(
                f"{BACKEND_URL}/query",
                data={
                    "filename": st.session_state.pdf_name,
                    "query": prompt
                },
                stream=True,
                timeout=60
            )
            response.raise_for_status()

            full_response = st.write_stream(iter_response_chunks(response))

        except Exception as exc:
            full_response = f"Sorry, I could not stream the response. Error: {exc}"
            st.error(full_response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )