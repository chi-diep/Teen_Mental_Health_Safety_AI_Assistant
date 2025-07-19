# app/rag_chain.py

import json
import os
import hashlib
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.retrievers.multi_query import MultiQueryRetriever

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_PATH = os.path.join(BASE_DIR, "data", "faq.json")
HOTLINE_PATH = os.path.join(BASE_DIR, "data", "hotlines.json")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
HASH_PATH = os.path.join(CHROMA_DIR, "source.hash")

def compute_data_hash():
    with open(FAQ_PATH, 'r', encoding='utf-8') as f1, open(HOTLINE_PATH, 'r', encoding='utf-8') as f2:
        raw = f1.read() + f2.read()
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()

def load_documents():
    def clean(text):
        return text.strip().replace('\n', ' ').replace('  ', ' ')

    with open(FAQ_PATH, 'r', encoding='utf-8') as f1, open(HOTLINE_PATH, 'r', encoding='utf-8') as f2:
        faq_data = json.load(f1)
        hotline_data = json.load(f2)

    seen = set()
    docs = []

    for item in faq_data:
        answer = clean(item['answer'])
        if answer not in seen:
            seen.add(answer)
            docs.append(Document(page_content=answer))

    for item in hotline_data:
        text = clean(f"{item.get('organization', '')} supports {item.get('topic', '')}. Call: {item.get('phone', '')}, Website: {item.get('website', '')}")
        if text not in seen:
            seen.add(text)
            docs.append(Document(page_content=text))

    return docs

def handle_statistical_question(question, all_data):
    q = question.lower()
    numbers = [v for entry in all_data for v in entry.values() if isinstance(v, (int, float))]

    if not numbers:
        return None

    if "total" in q or "sum" in q:
        return f"The total is {sum(numbers)}."
    elif "average" in q or "mean" in q:
        return f"The average is {sum(numbers) / len(numbers):.2f}."
    elif "maximum" in q or "max" in q:
        return f"The maximum value is {max(numbers)}."
    elif "minimum" in q or "min" in q:
        return f"The minimum value is {min(numbers)}."
    elif "count" in q or "how many" in q:
        return f"There are {len(all_data)} records in total."

    return None

rag_chain = None

def build_rag_chain():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = load_documents()
    current_hash = compute_data_hash()

    hash_mismatch = True
    if os.path.exists(HASH_PATH):
        with open(HASH_PATH, 'r') as f:
            saved_hash = f.read().strip()
        hash_mismatch = current_hash != saved_hash

    if not hash_mismatch and os.path.exists(os.path.join(CHROMA_DIR, "index")):
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
        with open(HASH_PATH, 'w') as f:
            f.write(current_hash)

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
        llm=OllamaLLM(model="tinyllama")
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are Guardiané, a safety and wellness assistant for parents of teens.

Answer the following question using only the context provided. If the context does not contain an answer, respond with:
"I'm sorry, I could not find information related to your question."

<context>
{context}
</context>

Question: {question}
Answer:"""
    )

    qa_chain = load_qa_chain(
        llm=OllamaLLM(model="tinyllama"),
        chain_type="stuff",
        prompt=prompt
    )

    return lambda question: qa_chain.run(input_documents=retriever.get_relevant_documents(question), question=question)

def hybrid_qa(question: str) -> str:
    global rag_chain
    try:
        with open(FAQ_PATH, 'r', encoding='utf-8') as f1, open(HOTLINE_PATH, 'r', encoding='utf-8') as f2:
            all_data = json.load(f1) + json.load(f2)

        stat_answer = handle_statistical_question(question, all_data)
        if stat_answer:
            return stat_answer

        greetings = ["hello", "hi", "hey", "what's up", "how are you", "good morning", "good evening"]
        if any(greet in question.lower() for greet in greetings):
            return "Hi there! I'm Guardiané. I help with teen mental health, safety, and wellness. Ask me anything specific."

        if rag_chain is None:
            rag_chain = build_rag_chain()

        result = rag_chain(question).strip()
        return result

    except Exception as e:
        return f"An error occurred while processing your question: {e}"
