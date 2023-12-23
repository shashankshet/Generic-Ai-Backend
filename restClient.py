from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
import textwrap
import warnings
import uvicorn
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with the specific origins you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Suppress all warnings
warnings.filterwarnings("ignore")

# Load the Hugging Face model only once
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.8, "max_length": 512})
chain = load_qa_chain(llm, chain_type="stuff")

# Load text document
loader = TextLoader("data.txt")
document = loader.load()

# Text Splitter
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
docs = text_splitter.split_documents(document)

# Embeddings and FAISS
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)


class Question(BaseModel):
    question: str


@app.post("/ask")
async def get_answer(question: Question = Body(...)):
    # Process question and get similarity search
    doc_result = db.similarity_search(question.question)

    # Get answer using the Q-A chain
    answer = chain.run(input_documents=doc_result, question=question.question)

    return {"answer": answer}
    
@app.get("/health")
async def health():
    try:
        return {"health":"ok"}
    except Exception as e:
        raise e

# Run the FastAPI application
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
