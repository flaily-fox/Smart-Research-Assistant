# The Main Streamlit Application

# This file will handle the main Streamlit app logic, including UI and interaction with core functions
 
# app.py
import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Import functions from our new core modules
from core.document_processor import (
    extract_text_from_pdf,
    extract_text_from_txt,
    get_text_chunks,
    get_embeddings,
    find_relevant_chunks # find_relevant_chunks is used by llm_functions, but also sometimes directly for debugging
)
from core.llm_functions import (
    get_document_summary,
    answer_question_with_rag,
    generate_challenge_questions,
    evaluate_user_answer
)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Google Generative AI
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("GEMINI_API_KEY not found in .env file. Please set it up.")
    st.stop() # Stop execution if API key is missing

# Initialize Gemini Models (these instances will be passed to core functions)
generation_model = genai.GenerativeModel('models/gemini-1.5-flash-latest') # Or 'models/gemini-1.5-pro-latest'
embedding_model_name = 'models/embedding-001' # String name for embedding model

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Smart Research Assistant")
st.title("📚 Smart Research Assistant")

# --- Session State Initialization ---
if "document_text" not in st.session_state:
    st.session_state.document_text = None
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []
if "chunk_embeddings" not in st.session_state:
    st.session_state.chunk_embeddings = []
if "document_summary" not in st.session_state:
    st.session_state.document_summary = "Awaiting document upload." # More descriptive initial message
if "challenge_questions" not in st.session_state:
    st.session_state.challenge_questions = []
if "challenge_answers" not in st.session_state:
    st.session_state.challenge_answers = []
if "challenge_evaluations" not in st.session_state:
    st.session_state.challenge_evaluations = []
if "uploaded_file_id" not in st.session_state:
    st.session_state.uploaded_file_id = None
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "embeddings_generated" not in st.session_state:
    st.session_state.embeddings_generated = False
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- UI for Document Upload ---
st.header("Upload Your Document")
uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
    # Check if a new file is uploaded or if the session state was cleared
    if "uploaded_file_id" not in st.session_state or st.session_state.uploaded_file_id != uploaded_file.file_id:
        st.session_state.uploaded_file_id = uploaded_file.file_id
        
        # Reset relevant session states for a new upload
        st.session_state.document_text = None
        st.session_state.text_chunks = []
        st.session_state.chunk_embeddings = []
        st.session_state.document_summary = "Awaiting summary generation..."
        st.session_state.file_processed = False
        st.session_state.embeddings_generated = False
        st.session_state.messages = [] # Clear chat history for new document
        st.session_state.challenge_questions = []
        st.session_state.challenge_answers = []
        st.session_state.challenge_evaluations = []

        # --- Extract Text ---
        with st.spinner("Extracting text from document..."):
            try:
                if uploaded_file.type == "application/pdf":
                    document_content = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "text/plain":
                    document_content = extract_text_from_txt(uploaded_file)
                else:
                    st.error("Unsupported file type. Please upload a PDF or TXT file.")
                    document_content = None # Ensure it's None if unsupported

                if document_content:
                    st.session_state.document_text = document_content
                    st.success("Document uploaded and text extracted successfully!")
                    st.write(f"Document contains {len(document_content)} characters.")
                    st.session_state.file_processed = True
                else:
                    st.error("Failed to extract text from document. Please ensure it's a valid PDF/TXT.")
                    st.session_state.file_processed = False

            except Exception as e:
                st.error(f"An unexpected error occurred during text extraction: {e}")
                st.session_state.file_processed = False

    # Proceed with chunking and embeddings ONLY if text was successfully extracted and embeddings not yet generated
    if st.session_state.document_text and not st.session_state.embeddings_generated:
        with st.spinner("Splitting document into chunks and generating embeddings (this might take a moment for large files)..."):
            try:
                text_chunks = get_text_chunks(st.session_state.document_text) # Call from document_processor
                chunk_embeddings = get_embeddings(text_chunks, embedding_model_name) # Call from document_processor

                st.session_state.text_chunks = text_chunks
                st.session_state.chunk_embeddings = chunk_embeddings
                st.session_state.embeddings_generated = True

                st.success(f"Generated {len(st.session_state.text_chunks)} text chunks and {len(st.session_state.chunk_embeddings)} embeddings.")

            except Exception as e:
                st.error(f"Error during chunking or embedding generation: {e}. Please try another file.")
                st.session_state.embeddings_generated = False
                # Clear relevant states to allow re-processing on next attempt
                st.session_state.document_text = None
                st.session_state.text_chunks = []
                st.session_state.chunk_embeddings = []

    # --- Auto Summary (After chunks/embeddings are ready) ---
    # Only generate summary if document is processed, embeddings are ready, and summary hasn't been generated yet for THIS document.
    if st.session_state.file_processed and st.session_state.embeddings_generated and st.session_state.document_summary == "Awaiting summary generation...":
        with st.spinner(f"Generating summary with {generation_model.model_name.replace('models/', '')}..."):
            summary_text = get_document_summary(st.session_state.document_text, generation_model) # Call from llm_functions
            st.session_state.document_summary = summary_text
            if summary_text and "Error" not in summary_text and "Could not generate summary." not in summary_text:
                st.success("Summary generated!")
            else:
                st.warning("Could not generate summary. Check API key/model configuration or document content.")
                st.session_state.document_summary = "Failed to generate summary." # Update status


# --- Display Summary ---
if st.session_state.document_summary:
    st.header("Document Summary")
    st.write(st.session_state.document_summary)


# --- Interaction Modes (Enabled only if document is uploaded and processed) ---
if st.session_state.document_text and st.session_state.text_chunks and st.session_state.chunk_embeddings:
    st.header("Interact with Your Document")
    mode = st.radio("Choose interaction mode:", ("Ask Anything", "Challenge Me"), key="interaction_mode")

    if mode == "Ask Anything":
        st.subheader("Ask Me Anything about the document!")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "justification" in message and message["justification"]:
                    with st.expander("Show Justification"): # Expander for justification
                        st.caption(message["justification"]) # Display inside the expander
                    
                    
                if "snippets" in message and message["snippets"]:
                    with st.expander("Show supporting snippets"):
                        for i, snippet in enumerate(message["snippets"]):
                            st.text_area(f"Snippet {i+1}", snippet, height=150, key=f"chat_snippet_{message['turn']}_{i}")

        if prompt := st.chat_input("Your question:", key="chat_input"):
            if not GEMINI_API_KEY:
                st.warning("Please provide a GEMINI_API_KEY in the .env file to use this feature.")
            elif not st.session_state.embeddings_generated:
                 st.warning("Please upload and process a document first to ask questions.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt, "turn": len(st.session_state.messages)})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.spinner("Searching document and generating answer..."):
                    answer, justification, snippets_to_highlight = answer_question_with_rag(
                        prompt,
                        st.session_state.document_text,
                        st.session_state.text_chunks,
                        st.session_state.chunk_embeddings,
                        generation_model, # Pass the model instance
                        embedding_model_name # Pass the model name
                    )
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "justification": justification,
                    "snippets": snippets_to_highlight,
                    "turn": len(st.session_state.messages)
                })

                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander("Show Justification"):
                        st.caption(justification)
                    if snippets_to_highlight:
                        with st.expander("Show supporting snippets"):
                            for i, snippet in enumerate(snippets_to_highlight):
                                st.text_area(f"Snippet {i+1}", snippet, height=150, key=f"chat_snippet_new_{len(st.session_state.messages)-1}_{i}")

    elif mode == "Challenge Me":
        st.subheader("Challenge Yourself!")

        if st.button("Generate New Questions", key="generate_questions_button"):
            if not GEMINI_API_KEY:
                st.warning("Please provide a GEMINI_API_KEY in the .env file to use this feature.")
            else:
                with st.spinner("Generating new challenge questions..."):
                    st.session_state.challenge_questions = generate_challenge_questions(
                        st.session_state.document_text,
                        generation_model # Pass the model instance
                    )
                    st.session_state.challenge_answers = [""] * len(st.session_state.challenge_questions)
                    st.session_state.challenge_evaluations = [""] * len(st.session_state.challenge_questions)
                if not st.session_state.challenge_questions:
                    st.error("Failed to generate questions. Please try again or check document content.")

        if st.session_state.get("challenge_questions"):
            st.write("Answer the following questions based on the document:")
            for i, question_text in enumerate(st.session_state.challenge_questions):
                st.markdown(f"**Q{i+1}: {question_text}**")
                user_challenge_answer = st.text_area(f"Your answer for Q{i+1}:",
                                                     value=st.session_state.challenge_answers[i],
                                                     key=f"challenge_answer_{i}")
                st.session_state.challenge_answers[i] = user_challenge_answer

                if st.button(f"Submit Answer for Q{i+1}", key=f"submit_challenge_{i}"):
                    if user_challenge_answer.strip():
                        with st.spinner("Evaluating your answer..."):
                            evaluation, context_used = evaluate_user_answer(
                                question_text,
                                user_challenge_answer,
                                st.session_state.document_text,
                                st.session_state.text_chunks,
                                st.session_state.chunk_embeddings,
                                generation_model, # Pass the model instance
                                embedding_model_name # Pass the model name
                            )
                        st.session_state.challenge_evaluations[i] = evaluation
                        st.write(f"**Evaluation for Q{i+1}:**")
                        with st.expander("Show detailed evaluation"): # Expander for evaluation
                            st.write(evaluation) # Display inside the expander
                        with st.expander(f"Show context used for evaluation of Q{i+1}"):
                            st.text_area(f"Context used for Q{i+1}", context_used, height=200, key=f"eval_context_{i}")
                    else:
                        st.warning("Please provide an answer before submitting.")
                        
                
                if st.session_state.challenge_evaluations[i]:
                    st.write(f"**Previous Evaluation for Q{i+1}:**")
                    with st.expander("Show previous detailed evaluation"):
                        st.write(st.session_state.challenge_evaluations[i])
                st.markdown("---")

        else:
            st.info("Click 'Generate New Questions' to start the challenge!")

else:
    st.warning("Please upload a document to enable interaction modes.")
    if st.session_state.document_text is None and uploaded_file is not None:
        st.warning("Document processing failed or is incomplete. Please check for errors above.")

st.markdown("---")
