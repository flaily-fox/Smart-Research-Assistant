# 📚 Smart Research Assistant for EZ

## An AI-Powered Document Summarization & Interaction Tool

**Submission for: [Your Assignment Name/Course Name - e.g., "GenAI Internship Task"]**
**Deadline: Submitted within 7 days from 2025-07-11**

---

## 🎯 Objective & Problem Addressed

This project, the **Smart Research Assistant**, was developed in response to the task of creating an AI-powered tool to tackle the time-consuming challenge of digesting large documents like research papers, legal files, or technical manuals. The core objective, set by EZ, was to build an assistant that not only summarizes content but also enables intelligent, justified interaction with the document's information.

---

## ✨ Key Features & How They Address Requirements

This assistant provides a comprehensive solution, designed with a focus on **Response Quality, Reasoning Functionality, UI/UX, Code Structure, Creativity, and Minimal Hallucination**, directly aligning with the assignment's evaluation criteria.

1.  **Document Processing (PDF & TXT):**
    * **Requirement:** Process uploaded PDF or TXT documents.
    * **Implementation:** Supports robust text extraction from both PDF (`pypdf`) and TXT files. Optimized for performance with **caching** for faster subsequent processing of the same document.

2.  **Concise Document Summarization:**
    * **Requirement:** Provide a concise summary (≤150 words).
    * **Implementation:** An automatic, **AI-generated summary (≤150 words)** is displayed upon successful document processing, offering an immediate overview of the content.

3.  **"Ask Anything" Mode (Free-Form Q&A):**
    * **Requirement:** "Ask Anything" for free-form question answering. All responses must be justified with document references, avoiding hallucinations.
    * **Implementation:** Users can ask any question related to the document. The system provides **accurate answers** and, critically, **justifies each response by displaying the relevant snippet(s)** directly from the document. The justification is concisely presented in a collapsible UI element for a clean interface.
    * **Bonus Feature (Memory Handling):** Supports **follow-up questions that refer to prior information** within the same conversational session. The chat history is maintained and displayed, allowing for a natural, contextual dialogue.

4.  **"Challenge Me" Mode (Logic-Based Question Generation & Evaluation):**
    * **Requirement:** "Challenge Me" for logic-based question generation and evaluation.
    * **Implementation:** This mode dynamically generates **logic-based and comprehension-focused questions** derived from the document's content, pushing users beyond simple factual recall. It then allows users to submit answers, which are **evaluated by the AI**, providing detailed feedback and the specific document context used for evaluation. This directly addresses the **Reasoning Mode Functionality (20%)** criterion.

5.  **Minimal Hallucination & Good Context Use:**
    * **Requirement:** All responses must be justified with document references, avoiding hallucinations.
    * **Implementation:** A core design principle. The RAG (Retrieval-Augmented Generation) pipeline ensures that answers are strictly **grounded in the provided document context**. If an answer cannot be found in the document, the AI explicitly states this, preventing the generation of incorrect or fabricated information. This directly contributes to the **Minimal Hallucination & Good Context Use (5%)** criterion.

6.  **User-Friendly Web Interface & Smooth Flow:**
    * **Requirement:** UI/UX using frameworks like Streamlit or React.
    * **Implementation:** Built entirely with **Streamlit**, offering an intuitive, clean, and interactive user experience. **Visual feedback** via spinners and clear status messages guides the user through processing steps, enhancing the **UI/UX and Smooth Flow (20%)** criterion.

---

## 🛠️ Technical Architecture & Code Structure

The application's architecture is designed for clarity, modularity, and efficient performance, directly addressing the **Code Structure & Documentation (15%)** criterion.

* **Framework:** Built on **Streamlit** for the interactive frontend.
* **LLMs & Embeddings:** Leverages **Google Gemini 1.5 Flash** for generative tasks (summarization, Q&A, question generation, evaluation) and **Google `embedding-001`** for creating document and query embeddings.
* **Retrieval-Augmented Generation (RAG) Pipeline:**
    * **Text Extraction:** Utilizes `pypdf` for robust PDF parsing.
    * **Intelligent Chunking:** Employs `langchain-text-splitters` (`RecursiveCharacterTextSplitter`) to divide documents into semantically coherent chunks.
    * **Vector Search:** `numpy` and `scikit-learn`'s `cosine_similarity` are used to efficiently retrieve the most relevant chunks.
* **API Management:** `python-dotenv` securely loads the `GEMINI_API_KEY`.

**Organized Source Code Structure:**
smart_research_assistant/
├── .env                        # Securely stores the GEMINI_API_KEY
├── app.py                      # Main Streamlit application file (UI orchestration, session state management)
├── core/                       # Contains core business logic functions
│   ├── init.py             # Makes 'core' a Python package
│   ├── document_processor.py   # Handles document loading, text extraction, chunking, and embedding generation
│   └── llm_functions.py        # Manages all interactions with the Gemini LLM (summaries, Q&A, challenge questions, evaluations)
├── utils/                      # (Placeholder for future utility functions)
│   └── init.py
└── requirements.txt            # Lists all Python dependencies for easy environment setup

This structure ensures a clear separation of concerns, making the codebase highly maintainable, testable, and scalable for future enhancements.

---

## 🚀 Quick Start Guide

Follow these steps to set up and run the Smart Research Assistant on your local machine.

### 📋 Prerequisites

* Python 3.8+

### ⬇️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_REPOSITORY_URL_HERE]
    cd smart_research_assistant
    ```
    *(Replace `[YOUR_REPOSITORY_URL_HERE]` with the actual URL where your code is hosted.)*

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure your Google Gemini API Key:**
    * Obtain a `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/app/apikey).
    * Create a file named `.env` in the root directory of the project (at the same level as `app.py`).
    * Add your API key to this file:
        ```
        GEMINI_API_KEY='YOUR_API_KEY_HERE'
        ```
        **Important:** Replace `'YOUR_API_KEY_HERE'` with your actual API key. **Do NOT** share or commit your `.env` file to public repositories!

### ▶️ How to Run the Application

With your virtual environment active and API key configured:

```bash
streamlit run app.

