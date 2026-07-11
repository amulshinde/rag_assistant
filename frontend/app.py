import streamlit as st
import requests

# 1. Force the app to use the full width of the screen
st.set_page_config(
    page_title="PDF RAG Application",
    layout="wide"
)

# ---------------------------------------------------------
# SIDEBAR SECTION: PDF Upload & Status
# ---------------------------------------------------------
with st.sidebar:
    st.header("📄 Document Center")
    uploaded_file = st.file_uploader(
        "Upload your PDF document", 
        type=["pdf"],
        help="Upload a PDF to build or load the vector index."
    )
    
    if uploaded_file:
        st.success(f"✓ {uploaded_file.name} loaded")
    else:
        st.info("💡 Please upload a PDF to begin.")

# ---------------------------------------------------------
# MAIN SECTION: Query & Answer Interaction
# ---------------------------------------------------------
st.title("🤖 PDF Assistant")
st.write("Ask questions directly against your uploaded document.")

# Input text box spanning the full main width
query = st.text_input("Ask Question:", placeholder="e.g., What is the summary of Chapter 1?")

if st.button("Submit Query", type="primary"):
    if not uploaded_file:
        st.error("Please upload a PDF file in the sidebar first!")
    elif not query.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Searching document and generating answer..."):
            try:
                # Prepare payload for the backend API
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                data = {"query": query}
                
                # Point to your backend service running on the internal docker network
                response = requests.post("http://backend:8000/query", files=files, data=data)
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer found.")
                    
                    st.subheader("💡 Answer from Gemini")
                    st.info(answer)
                else:
                    st.error(f"Backend Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")