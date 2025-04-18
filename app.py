import os
import fitz  # PyMuPDF
from flask import Flask, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Environment Variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

# Temporary memory store
chat_memory_store = {}

# Allowed file extensions and max file size (5MB for example)
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

# Check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Limit file size (max 5 MB)
def is_file_size_allowed(file):
    return len(file.read()) <= MAX_FILE_SIZE

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

@app.route('/', methods=['GET'])
def home():
    return "✅ CV Assistant API is running!"

@app.route('/upload', methods=['POST'])
def upload_pdf():
    file = request.files.get("file")
    user_id = request.form.get("user_id", "default_user")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Check file size
    if not is_file_size_allowed(file):
        return jsonify({"error": f"File is too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)} MB."}), 400

    # Check file extension
    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return jsonify({"error": "Invalid file type. Only PDF files are allowed."}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(file_path)
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        return jsonify({"error": "Error saving file."}), 500

    text = extract_text_from_pdf(file_path)

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", ", "]
    )
    chunks = text_splitter.split_text(text)

    # Create FAISS vector store
    vector_store = FAISS.from_texts(chunks, embedding_model)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Set up the Groq chat model and memory
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0.7)
    memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True)

    # Create ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    chat_memory_store[user_id] = qa_chain
    logger.info(f"CV uploaded and processed successfully for user {user_id}.")
    return jsonify({"message": "CV uploaded and processed successfully", "user_id": user_id})

@app.route('/chat', methods=['POST'])
def chat():
    user_id = request.json.get("user_id", "default_user")
    question = request.json.get("question")

    if user_id not in chat_memory_store:
        return jsonify({"error": "Please upload a CV first"}), 400

    qa_chain = chat_memory_store[user_id]
    result = qa_chain({"question": question, "chat_history": qa_chain.memory.chat_memory})

    return jsonify({"answer": result["answer"]})

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
