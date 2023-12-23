from langchain.document_loaders import TextLoader
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


os.environ['HUGGINGFACEHUB_API_TOKEN']


loader = TextLoader("data.txt")
document = loader.load()


# pre-processing (remove /n)
import textwrap


def wrap_text_preserver_newline(text: str, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_lines = '\n'.join(wrapped_lines)
    return wrapped_lines


# Text Splitter
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
docs = text_splitter.split_documents(document)

# embedding

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)

query = "What is car?"


doc = db.similarity_search(query)

# print(wrap_text_preserver_newline(str(doc[0].page_content)))

#Q-A

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub

llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temparature":0.8,"max_length":512})
chain = load_qa_chain(llm,chain_type="stuff")

query_txt= input("Enter your question: ")
docResult = db.similarity_search(query_txt)


print(f"Answer: {chain.run(input_documents = docResult, question=query_txt)}")
