
import google.generativeai as genai
from core.document_processor import find_relevant_chunks # Import RAG function

# LLM model configuration will be passed from app.py
# generation_model will be an instance of genai.GenerativeModel
# embedding_model_name will be the string name 'models/embedding-001'

# --- Function for LLM-based summarization ---
# @st.cache_data removed here; caching should be handled by the caller (app.py)
def get_document_summary(document_text, generation_model):
    if not document_text:
        return "No document text available for summarization."

    try:
        prompt = f"Summarize the following document concisely in less than 150 words. Focus on the main points and overall topic:\n\n{document_text[:10000]}..."

        response = generation_model.generate_content(prompt)
        summary = response.text
        if len(summary.split()) > 150:
            summary_words = summary.split()[:150]
            summary = " ".join(summary_words) + "..."
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}") # Use print for core module logs
        return "Could not generate summary."

# --- Function for LLM-based Question Answering (Ask Anything) ---
def answer_question_with_rag(question, document_text, text_chunks, chunk_embeddings, generation_model, embedding_model_name):
    if not document_text:
        return "Please upload a document first.", "No document available.", []

    relevant_chunks, justifications, highlight_snippets = find_relevant_chunks(
        question, text_chunks, chunk_embeddings, embedding_model_name, top_k=5
    )

    if not relevant_chunks:
        return "Could not find relevant information in the document.", "No relevant chunks found.", []

    context = "\n\n".join(relevant_chunks)

    prompt = f"""
    You are a helpful assistant. Answer the following question ONLY based on the provided context from the document.
    Do not make up any information. If the answer cannot be found in the context, state "The information is not directly available in the provided document context."

    Question: {question}

    Context from document:
    {context}

    Your Answer:
    """

    try:
        response = generation_model.generate_content(prompt)
        answer = response.text
        justification_str = "Based on: " + "; ".join(justifications)
        return answer, justification_str, highlight_snippets
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Error generating answer.", "An error occurred during generation.", []

# --- Functions for Challenge Me Mode ---
def generate_challenge_questions(document_text, generation_model, num_questions=3):
    if not document_text:
        return []

    context_for_questions = document_text[:15000]

    prompt = f"""
    Based on the following document, generate {num_questions} unique, logic-based or comprehension-focused questions.
    These questions should require understanding and reasoning beyond simple fact retrieval.
    Each question should be answerable from the document.
    Format each question as "Q[Number]: [Question Text]".

    Document Context:
    {context_for_questions}

    Questions:
    """
    try:
        response = generation_model.generate_content(prompt)
        questions_raw = response.text
        parsed_questions = []
        for line in questions_raw.split('\n'):
            line = line.strip()
            if line.startswith("Q") and ":" in line:
                q_text = line.split(":", 1)[1].strip()
                if q_text:
                    parsed_questions.append(q_text)
            if len(parsed_questions) >= num_questions:
                break
        return parsed_questions[:num_questions]
    except Exception as e:
        print(f"Error generating challenge questions: {e}")
        return []

def evaluate_user_answer(question, user_answer, document_text, text_chunks, chunk_embeddings, generation_model, embedding_model_name):
    relevant_chunks, _, _ = find_relevant_chunks(question, text_chunks, chunk_embeddings, embedding_model_name, top_k=5)
    if not relevant_chunks:
        return "Could not find relevant document context for evaluation.", "N/A"

    context = "\n\n".join(relevant_chunks)

    prompt = f"""
    You are an AI assistant tasked with evaluating a user's answer to a question based on a document.
    Your goal is to determine if the user's answer is correct, partially correct, or incorrect,
    and provide justification by referencing the provided document context.

    Question: {question}
    User's Answer: {user_answer}

    Document Context:
    {context}

    Evaluate the user's answer and provide feedback based ONLY on the document context.
    State clearly if the answer is Correct, Partially Correct, or Incorrect.
    Then, provide a brief justification explaining why, citing information from the document.

    Example Response Format:
    Evaluation: Correct/Partially Correct/Incorrect
    Justification: [Explanation based on document context]
    """
    try:
        response = generation_model.generate_content(prompt)
        evaluation_result = response.text
        return evaluation_result, context
    except Exception as e:
        print(f"Error evaluating answer: {e}")
        return "Error during evaluation.", "N/A"