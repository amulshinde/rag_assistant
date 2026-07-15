import streamlit as st
import requests

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="PDF RAG Assistant",
    layout="wide"
)

BACKEND_URL = "http://backend:8000"

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:

    st.header("📄 Document Center")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file:

        st.success(f"✓ {uploaded_file.name}")

        # New document uploaded
        if uploaded_file.name != st.session_state.uploaded_filename:

            with st.spinner("Generating Embeddings..."):

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
                        st.session_state.answer = ""
                        st.session_state.last_query = ""

                        st.success("Embeddings Ready ✅")

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

st.write("Ask questions about your uploaded PDF.")

# ---------------------------------------------------------
# Display Latest Answer
# ---------------------------------------------------------

if st.session_state.answer:

    st.subheader("💡 Answer")

    st.info(st.session_state.answer)

    st.caption(
        f"Question : {st.session_state.last_query}"
    )

st.markdown("---")

# ---------------------------------------------------------
# Question Box
# ---------------------------------------------------------

query = st.text_input(
    "Ask Question",
    placeholder="e.g. Summarize Chapter 2"
)

submit = st.button(
    "Submit Query",
    type="primary",
    use_container_width=True
)

# ---------------------------------------------------------
# Query
# ---------------------------------------------------------

if submit:

    if st.session_state.pdf_name is None:

        st.error("Please upload a PDF first.")

    elif not query.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching..."):

            try:

                data = {
                    "filename": st.session_state.pdf_name,
                    "query": query
                }

                response = requests.post(
                    f"{BACKEND_URL}/query",
                    data=data
                )

                if response.status_code == 200:

                    st.session_state.answer = response.json()["answer"]
                    st.session_state.last_query = query

                    st.rerun()

                else:

                    st.error(response.text)

            except Exception as e:

                st.error(e)