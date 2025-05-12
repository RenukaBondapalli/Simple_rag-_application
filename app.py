from flask import Flask, request, render_template_string
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from werkzeug.utils import secure_filename
import os

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyA7YdIMRXPIlHfPSsn3vN3ZkiffBQhhEy0"  # Replace with your real Gemini Flash API key

# Config
UPLOAD_FOLDER = "uploads"
CHROMA_DB_DIR = "./chroma_db"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load embedding model once
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG Web App</title>
</head>
<body>
    <h2>Upload a .txt file and ask a question</h2>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file" required><br><br>
        <input type="text" name="question" placeholder="Ask a question..." size="50" required><br><br>
        <input type="submit" value="Submit">
    </form>
    {% if answer %}
        <h3>Answer:</h3>
        <p>{{ answer }}</p>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    if request.method == "POST":
        file = request.files["file"]
        question = request.form["question"]
        if file and question:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Load and split
            loader = TextLoader(filepath)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = splitter.split_documents(documents)

            # Vector DB
            vectordb = Chroma.from_documents(docs, embedding, persist_directory=CHROMA_DB_DIR)

            # Gemini LLM
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

            # RAG
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
            answer = qa.run(question)

    return render_template_string(HTML_TEMPLATE, answer=answer)

if __name__ == "__main__":
    app.run()
