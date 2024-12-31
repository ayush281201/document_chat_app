from flask import Flask, render_template, request, jsonify
import os
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from constants import CHROMA_SETTINGS
from ingest import does_vectorstore_exist, load_single_document

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'csv', 'docx', 'html', 'md', 'epub', 'eml'}

HOST = "0.0.0.0"
PORT = 5000

persist_directory = os.environ.get('PERSIST_DIRECTORY', 'db')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME', 'all-MiniLM-L6-v2')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        process_uploaded_file(filename)
        return render_template('index.html', message = 'File uploaded successfully')
    else:
        return jsonify({'error': 'File not allowed'}), 400

def process_uploaded_file(file_path):
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    if not does_vectorstore_exist(persist_directory):
        print("Creating new vectorstore")
        db = Chroma.from_documents(load_single_document(file_path), embeddings, persist_directory=persist_directory)
    else:
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    db.persist()

@app.route('/chat', methods=['POST'])
def chat():
    query = request.form.get('query')
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    llm = Ollama(model="mistral")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    result = qa(query)
    answer = result['result']
    source_docs = result.get('source_documents', [])
    response = {'answer': answer, 'source_docs': [{'source': doc.metadata['source'], 'content': doc.page_content} for doc in source_docs]}
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, host=HOST, port=PORT)