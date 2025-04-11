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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

# Load Environment Variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

# Temporary memory store
chat_memory_store = {}

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

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    text = extract_text_from_pdf(file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", ", "]
    )
    chunks = text_splitter.split_text(text)

    vector_store = FAISS.from_texts(chunks, embedding_model)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0.7)
    memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    chat_memory_store[user_id] = qa_chain
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
    app.run(host='0.0.0.0', port=8000)
