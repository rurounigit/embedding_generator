# Document Embedding & FAISS Index Generator (Gradio App)

This Gradio application provides a user-friendly interface to process text documents (`.txt` files), generate embeddings using Google's Generative AI, and create a FAISS vector index. The output is a downloadable `.zip` file containing the `index.faiss` and `index.pkl` files, ready to be used by a Retrieval Augmented Generation (RAG) system, such as the companion "Angela Han AI" Discord bot.

## ✨ Features

*   **File Upload:** Allows uploading multiple `.txt` files (e.g., Whisper transcripts).
*   **Document Processing:**
    *   Loads text content from uploaded files.
    *   Splits documents into manageable chunks using `RecursiveCharacterTextSplitter`.
*   **Embedding Generation:** Uses Google's `text-embedding-004` model (configurable) via `langchain_google_genai` to create embeddings for the document chunks.
*   **FAISS Index Creation:** Builds a FAISS vector store from the embedded chunks.
*   **Output:** Generates a `faiss_index_google.zip` file containing:
    *   `index.faiss`: The FAISS index.
    *   `index.pkl`: The LangChain FAISS docstore and index_to_docstore_id mapping.
*   **User-Friendly Interface:** Built with Gradio for easy interaction.
*   **Status Updates:** Provides feedback on the processing status.
*   **Configuration:** Requires a `GOOGLE_API_KEY` set as an environment variable or Hugging Face Space Secret.

## 🛠️ Technologies Used

*   **Python 3.8+**
*   **Gradio:** For creating the web UI.
*   **LangChain:** Framework for LLM application development.
    *   `langchain_community.document_loaders`: For loading text files.
    *   `langchain.text_splitter`: For splitting documents.
    *   `langchain_community.vectorstores.FAISS`: For FAISS vector store operations.
    *   `langchain_google_genai`: For Google Generative AI Embeddings.
*   **Google Generative AI:**
    *   Embedding Model: `models/text-embedding-004` (or as configured).
*   **FAISS (faiss-cpu/faiss-gpu):** For efficient similarity search.
*   **python-dotenv:** For managing environment variables locally.
*   **Standard Python Libraries:** `os`, `tempfile`, `shutil`, `zipfile`, `traceback`.

## ⚙️ Prerequisites

1.  **Python 3.8 or higher.**
2.  **Google API Key:**
    *   Go to [Google AI Studio](https://aistudio.google.com/app/apikey) or Google Cloud Console.
    *   Create an API key.
    *   **Enable the "Generative Language API"** (also known as Gemini API) for your project.
    *   This key needs to be available as an environment variable (`GOOGLE_API_KEY`). When deploying to Hugging Face Spaces, set this as a "Secret".

## 🚀 Setup & Installation (Local Development)

1.  **Clone the repository (if applicable) or download `app.py` and `requirements.txt`:**
    ```bash
    # If in a repo:
    # git clone https://github.com/rurounigit/embedding_generator.git
    # cd your-repo-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Make sure you have `app.py` and `requirements.txt` in your current directory.
    ```bash
    pip install -r requirements.txt
    ```
    *(The provided `requirements.txt` includes `gradio`, `langchain`, `langchain-community`, `langchain-google-genai`, `faiss-cpu`, `python-dotenv`, `tiktoken`, and `unstructured`.)*

4.  **Configure Environment Variables (Local):**
    Create a `.env` file in the root directory and add your Google API key:
    ```env
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
    ```

## ▶️ Running the App

### Locally
To run the Gradio app locally:
```bash
python app.py
```
The application will typically be available at `http://127.0.0.1:7860` (Gradio will print the exact URL).

### On Hugging Face Spaces
1.  Create a new Space on Hugging Face.
2.  Choose "Gradio" as the SDK.
3.  Upload your `app.py` and `requirements.txt` files.
4.  Go to your Space's "Settings" tab.
5.  Under "Repository secrets," add a new secret:
    *   **Name:** `GOOGLE_API_KEY`
    *   **Value:** Your actual Google API Key
6.  The Space should build and launch your Gradio app.

## 💬 Usage

1.  **Open the Gradio App:** Access it via the local URL or your Hugging Face Space URL.
2.  **Upload Files:** Click on the "Upload Transcript Files (.txt)" area or drag and drop your `.txt` files. You can select multiple files.
3.  **Process:** Click the "Generate Embeddings & Index" button.
4.  **Monitor Status:** The "Status" box will show updates on the process (e.g., files being loaded, chunks created, index building).
5.  **Download Output:** Once processing is complete and successful, a download link/button for `faiss_index_google.zip` will appear under "Download Index Zip File". Click it to save the zip file.
    *   If an error occurs, an error message will be displayed in the "Status" box.

## 📁 Output

The application produces a zip file (typically named `faiss_index_google.zip`) containing:

*   `index.faiss`: The binary FAISS index file.
*   `index.pkl`: A Python pickle file containing the LangChain FAISS `docstore` (mapping from index IDs to document content and metadata) and `index_to_docstore_id` (mapping from FAISS index IDs to docstore IDs).

This zip file is structured to be directly usable by LangChain's `FAISS.load_local()` method when unzipped.

## ⚠️ Important Notes

*   **`GOOGLE_API_KEY`:** This is essential. The app will not function without a valid Google API key with the Generative Language API enabled.
*   **Embedding Model:** The app uses `models/text-embedding-004` by default. Ensure this matches the embedding model expected by any downstream RAG application (like the companion chatbot). The `models/` prefix is important for some Google API endpoints.
*   **`task_type="retrieval_document"`:** The embedding model is initialized with this task type, which is generally recommended for creating embeddings intended for document retrieval.
*   **Text File Encoding:** Ensure your `.txt` files are in a common encoding (like UTF-8) for best results with `TextLoader`.
*   **Chunking Parameters:** `chunk_size` (800) and `chunk_overlap` (180) in `RecursiveCharacterTextSplitter` can be adjusted in `app.py` if needed, depending on your content and embedding model limits.
*   **Resource Usage:** Processing many large files can be memory and CPU intensive.

## 🔧 Troubleshooting

*   **"ERROR: Embeddings model could not be initialized"**:
    *   Verify your `GOOGLE_API_KEY` is correct and set as an environment variable (local) or Space secret (Hugging Face).
    *   Ensure the Generative Language API is enabled for your Google Cloud project associated with the API key.
    *   Check your internet connection.
*   **"Could not load any text from the uploaded files"**:
    *   Ensure the uploaded files are indeed `.txt` files and contain text.
    *   Check for any special characters or encoding issues in the files.
*   **Errors during FAISS creation/saving**:
    *   This could be due to very large datasets exhausting memory. Try with fewer or smaller files.
    *   Rarely, issues with the `faiss-cpu` library installation.
*   **Hugging Face Space Build Failures:**
    *   Double-check `requirements.txt` for correct package names and compatibility.
    *   Look at the build logs in your Space for specific error messages.

## 📄 License

This project is provided as-is. If you are distributing it, consider adding a `LICENSE` file (e.g., MIT License).


---
title: Embedding Generator Gemini
emoji: 📈
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.27.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
