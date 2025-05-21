import gradio as gr
import os
import tempfile
import shutil
import zipfile
import traceback
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    print("CRITICAL ERROR: GOOGLE_API_KEY not found in environment variables. Cannot create embeddings.")
    # Optionally, raise an exception or exit if the key is absolutely required at startup
    # For Gradio, printing the error and letting the UI show failure is often better.

# --- Model Names ---
# MAKE SURE THIS MATCHES THE CHATBOT SPACE EXACTLY
GOOGLE_EMBEDDING_MODEL_NAME = "models/text-embedding-004" # <-- ADDED 'models/' PREFIX
print(f"Using Google Embedding Model: {GOOGLE_EMBEDDING_MODEL_NAME}")

# --- Initialize Embeddings ---
embeddings = None
if google_api_key:
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=GOOGLE_EMBEDDING_MODEL_NAME,
            google_api_key=google_api_key,
            # Specify "retrieval_document" task type for potentially better indexing performance
            task_type="retrieval_document"
        )
        print(f"Google AI Embeddings ({GOOGLE_EMBEDDING_MODEL_NAME}) initialized successfully.")
    except Exception as e:
        print(f"Error loading Google AI embeddings model '{GOOGLE_EMBEDDING_MODEL_NAME}': {e}")
        traceback.print_exc()
        # embeddings remains None
else:
    print("ERROR: Cannot initialize Google Embeddings without GOOGLE_API_KEY.")


# --- Core Function ---

def create_faiss_index_from_files(files_list):
    """Loads, splits, embeds files, creates FAISS index, saves it, and returns a zip file path."""
    if embeddings is None:
        return "ERROR: Embeddings model could not be initialized. Check API key and logs.", None
    if not files_list:
        return "Please upload transcript files first.", None

    print(f"Processing {len(files_list)} files...")
    temp_upload_dir = tempfile.TemporaryDirectory()
    temp_index_dir = tempfile.mkdtemp() # Directory to save FAISS index
    output_zip_path = None
    all_files_processed = True

    # 1. Copy uploaded files to a temporary directory
    for file_obj in files_list:
        # Use the original filename from the Gradio File object if available
        # Handle potential path issues if file_obj.name includes directories
        base_filename = os.path.basename(file_obj.name)
        temp_filepath = os.path.join(temp_upload_dir.name, base_filename)
        try:
            # Gradio file objects might be temp files already, shutil.copy is robust
            shutil.copy(file_obj.name, temp_filepath)
            print(f"Copied {base_filename} to {temp_filepath}")
        except Exception as e:
            print(f"Error copying file {file_obj.name}: {e}")
            traceback.print_exc()
            all_files_processed = False
            # Continue processing other files if possible

    if not all_files_processed:
        temp_upload_dir.cleanup()
        shutil.rmtree(temp_index_dir)
        return "Error handling one or more uploaded files during copy.", None

    try:
        # 2. Load documents from the temporary directory
        loader = DirectoryLoader(
            temp_upload_dir.name,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True,
            silent_errors=True, # Log errors but try to continue
            use_multithreading=True # Speed up loading if many files
        )
        documents = loader.load()

        if not documents:
            temp_upload_dir.cleanup()
            shutil.rmtree(temp_index_dir)
            print("Warning: No documents loaded from the temporary directory.")
            return "Could not load any text from the uploaded files. Ensure they are valid .txt files.", None

        print(f"Loaded {len(documents)} documents.")

        # 3. Split documents into chunks
        # Consider adjusting chunk_size/overlap based on embedding model and content
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=180)
        split_docs = text_splitter.split_documents(documents)
        print(f"Split into {len(split_docs)} chunks.")

        if not split_docs:
            temp_upload_dir.cleanup()
            shutil.rmtree(temp_index_dir)
            return "Error: No text chunks were generated after splitting the documents.", None

        # 4. Create FAISS vector store
        print(f"Creating FAISS index using Google Embeddings ({GOOGLE_EMBEDDING_MODEL_NAME})...")
        vector_store = FAISS.from_documents(split_docs, embeddings)
        print("FAISS index created successfully.")

        # 5. Save the FAISS index locally to the temp index directory
        index_save_path = os.path.join(temp_index_dir, "faiss_index")
        vector_store.save_local(index_save_path)
        print(f"FAISS index saved to: {index_save_path}")

        # Check if index files were created
        expected_faiss_file = os.path.join(index_save_path, "index.faiss")
        expected_pkl_file = os.path.join(index_save_path, "index.pkl")
        if not os.path.exists(expected_faiss_file) or not os.path.exists(expected_pkl_file):
             raise RuntimeError(f"FAISS index files not found at {index_save_path} after saving.")

        # 6. Zip the saved index files
        output_zip_filename = "faiss_index_google.zip"
        # Create zip in a directory Gradio can access for output
        # Using the same temp_index_dir might be okay, or create another temp file
        output_zip_path = os.path.join(tempfile.gettempdir(), output_zip_filename) # Put zip in standard temp

        with zipfile.ZipFile(output_zip_path, 'w') as zipf:
            zipf.write(expected_faiss_file, arcname="index.faiss")
            zipf.write(expected_pkl_file, arcname="index.pkl")
        print(f"Created zip file: {output_zip_path}")

        # 7. Cleanup temporary directories
        temp_upload_dir.cleanup()
        shutil.rmtree(temp_index_dir) # Remove the directory where index was saved before zipping

        return f"Successfully processed {len(files_list)} file(s). Index saved and zipped.", output_zip_path

    except Exception as e:
        # Cleanup in case of error
        temp_upload_dir.cleanup()
        if os.path.exists(temp_index_dir):
             shutil.rmtree(temp_index_dir)
        if output_zip_path and os.path.exists(output_zip_path):
             os.remove(output_zip_path) # Clean up partially created zip

        print(f"Error during processing: {e}")
        traceback.print_exc()
        return f"An error occurred during processing: {str(e)}", None

# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Document Embedding Generator 🧠💾")
    gr.Markdown(
        "Upload Whisper transcript `.txt` files. This tool will process them using Google Generative AI Embeddings "
        f"(`{GOOGLE_EMBEDDING_MODEL_NAME}`), create a FAISS vector index, and provide a downloadable `.zip` file "
        "containing `index.faiss` and `index.pkl`. This zip file can then be used in the companion RAG Chatbot Space."
        "\n\n**Requires `GOOGLE_API_KEY` to be set as a Secret.**"
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_uploader = gr.File(
                label="Upload Transcript Files (.txt)",
                file_count="multiple",
                file_types=[".txt"]
            )
            process_button = gr.Button("Generate Embeddings & Index")

        with gr.Column(scale=2):
            status_display = gr.Textbox(label="Status", interactive=False, lines=3)
            download_output = gr.File(label="Download Index Zip File", interactive=False) # Use gr.File for output

    # --- Wire Components ---
    process_button.click(
        fn=create_faiss_index_from_files,
        inputs=[file_uploader],
        outputs=[status_display, download_output]
    )

# --- Launch the App ---
if __name__ == "__main__":
    if not google_api_key:
        print("WARNING: GOOGLE_API_KEY not found. Embedding generation will fail.")
        # You might want the Gradio interface to load anyway to show the error message
    elif embeddings is None:
        print("ERROR: Google Embeddings failed to load during initial setup. Embedding generation will fail.")

    demo.queue()
    demo.launch()