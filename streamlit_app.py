"""Streamlit app for interacting with the Vidore RAG system."""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src import RAG

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Vidore RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS to make sidebar wider
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            width: 400px !important;
            min-width: 400px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "rag" not in st.session_state:
    st.session_state.rag = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "last_images" not in st.session_state:
    st.session_state.last_images = None
if "show_results" not in st.session_state:
    st.session_state.show_results = False

def initialize_rag():
    """Initialize the RAG system."""
    if st.session_state.rag is None:
        try:
            with st.spinner("Initializing RAG system..."):
                st.session_state.rag = RAG("vidore/colpali-v1.3")
            st.success("RAG system initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {e}")
            return False
    return True

def index_pdf_file(uploaded_file):
    """Index an uploaded PDF file."""
    if st.session_state.rag is None:
        st.error("RAG system not initialized")
        return False

    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = Path(tmp_file.name)

        with st.spinner(f"Indexing {uploaded_file.name}..."):
            st.session_state.rag.index_file(pdf_path=tmp_path, batch_size=1)

        # Clean up temporary file
        tmp_path.unlink()

        # Track indexed files
        if uploaded_file.name not in st.session_state.indexed_files:
            st.session_state.indexed_files.append(uploaded_file.name)

        st.success(f"Successfully indexed {uploaded_file.name}")
        return True

    except Exception as e:
        st.error(f"Failed to index {uploaded_file.name}: {e}")
        return False

def main():
    """Main Streamlit app."""
    st.title("📚 ViDoRe RAG Assistant")
    st.markdown("Upload PDF documents and ask questions about their content.")

    # Initialize RAG system
    if not initialize_rag():
        st.stop()

    # Sidebar for file management
    with st.sidebar:
        st.header("📁 Document Management")

        # File upload
        uploaded_files = st.file_uploader(
            "Upload PDF files to index",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or more PDF files to add to the knowledge base"
        )

        if uploaded_files:
            if st.button("Index Selected Files", type="primary"):
                progress_bar = st.progress(0)
                for i, uploaded_file in enumerate(uploaded_files):
                    index_pdf_file(uploaded_file)
                    progress_bar.progress((i + 1) / len(uploaded_files))
                progress_bar.empty()

        # Display indexed files
        if st.session_state.indexed_files:
            st.subheader("📋 Indexed Documents")
            for file_name in st.session_state.indexed_files:
                st.write(f"• {file_name}")

        # Clear index option
        if st.button("Clear Index", help="Remove all indexed documents"):
            if st.session_state.rag:
                # Note: This would require adding a clear method to the RAG class
                st.session_state.indexed_files = []
                st.success("Index cleared (note: this is a soft clear)")

    # Main content area - Chat section
    st.header("💬 Ask Questions")

    # Question input
    question = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="Ask anything about the indexed documents...",
        help="Type your question about the content of the uploaded PDFs"
    )

    # Answer parameters
    with st.expander("⚙️ Answer Settings"):
        top_k = st.slider("Number of documents to retrieve", 1, 10, 4)
        prefetch_limit = st.slider("Prefetch limit", 5, 50, 10)
        show_images = st.checkbox("Show retrieved images", value=False)

    # Ask button (positioned on the right)
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔍 Get Answer", type="primary", width='stretch'):
            if not question.strip():
                st.warning("Please enter a question.")
                st.session_state.show_results = False
            elif not st.session_state.indexed_files:
                st.warning("Please upload and index some PDF files first.")
                st.session_state.show_results = False
            else:
                with st.spinner("Searching documents and generating answer..."):
                    try:
                        if show_images:
                            response, images = st.session_state.rag.answer(
                                query=question,
                                top_k=top_k,
                                prefetch_limit=prefetch_limit,
                                with_images=True
                            )
                            st.session_state.last_images = images
                        else:
                            response = st.session_state.rag.answer(
                                query=question,
                                top_k=top_k,
                                prefetch_limit=prefetch_limit,
                                with_images=False
                            )
                            st.session_state.last_images = None

                        st.session_state.last_response = response
                        st.session_state.show_results = True
                        st.success("Answer generated!")

                    except Exception as e:
                        st.error(f"Error generating answer: {e}")
                        st.session_state.show_results = False

    # Display results outside the column layout (full width)
    if st.session_state.show_results and st.session_state.last_response:
        st.subheader("📝 Answer")
        st.write(st.session_state.last_response)

        # Display images if available
        if st.session_state.last_images:
            st.subheader("📸 Retrieved Document Pages")

            # Display images in a grid (2 columns)
            num_cols = 2
            cols = st.columns(num_cols)

            for i, image_data in enumerate(st.session_state.last_images):
                col_idx = i % num_cols
                with cols[col_idx]:
                    st.image(
                        f"data:image/jpeg;base64,{image_data['image']}",
                        caption=f"Page {image_data['page_no']} from {image_data['file']}",
                        width='stretch'
                    )

if __name__ == "__main__":
    main()
