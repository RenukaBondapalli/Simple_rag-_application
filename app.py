from flask import Flask, request, render_template_string
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
import os

# Set Gemini API key from environment
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "AIzaSyA7YdIMRXPIlHfPSsn3vN3ZkiffBQhhEy0")  # Replace or use env var

# Load document and build vector index ONCE
loader = TextLoader("dat.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")
vectordb = Chroma.from_documents(docs, embedding)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

# Flask app
app = Flask(__name__)

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Chat with Document</title>
    <style>
        body {
            background-color: #f4f6f8;
            font-family: Arial, sans-serif;
            padding: 20px;
        }
        .chatbox {
            max-width: 700px;
            margin: auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            padding: 20px;
        }
        h2 {
            color: #333;
        }
        .message {
            margin-bottom: 15px;
        }
        .user {
            color: #1a73e8;
            font-weight: bold;
        }
        .bot {
            color: #0b8043;
            font-weight: bold;
        }
        .response {
            background-color: #eef6f1;
            padding: 10px;
            border-radius: 8px;
        }
        input[type="text"] {
            width: 80%;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        input[type="submit"] {
            background-color: #1a73e8;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 5px;
            margin-left: 10px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="chatbox">
        <h2>📘 Ask a question about the document</h2>
        <form method="POST">
            <input type="text" name="question" placeholder="Type your question here..." required>
            <input type="submit" value="Ask">
        </form>
        {% if question and answer %}
            <div class="message"><span class="user">You:</span> {{ question }}</div>
            <div class="message"><span class="bot">Bot:</span> <div class="response">{{ answer }}</div></div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    question = None
    if request.method == "POST":
        question = request.form["question"]
        try:
            answer = qa.run(question)
        except Exception as e:
            answer = f"Error: {str(e)}"
    return render_template_string(HTML_TEMPLATE, answer=answer, question=question)

if __name__ == "__main__":
    app.run()
