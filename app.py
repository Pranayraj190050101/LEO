import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

FAISS_PATH = "data/faiss"

app = FastAPI()

print("Loading LEO...")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS.load_local(
    FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

SYSTEM_PROMPT = """You are LEO, an intelligent support assistant for Manulife's procurement team.
You help employees understand and resolve Fieldglass to Workday integration errors.

Use ONLY the context provided below to answer the question.
If the answer is not in the context, say: I don't have enough information to answer that. Please contact the Integration Support Team via ServiceNow.

Always structure your answer in 3 parts:
1. Likely Cause: explain what probably went wrong
2. Steps to Fix: clear step by step actions the user can take
3. Who to Contact: if self-service does not work, who should they escalate to

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join([
        "[Source: " + doc.metadata.get("source", "unknown") + "]\n" + doc.page_content
        for doc in docs
    ])

def get_sources(docs):
    sources = list(set([
        doc.metadata.get("source", "unknown")
        for doc in docs
    ]))
    return sources

class ChatRequest(BaseModel):
    question: str
    chat_history: List[dict] = []

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

@app.get("/")
def root():
    return {"status": "LEO is running"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    print(f"\nQuestion: {request.question}")

    # convert chat history to LangChain messages
    history = []
    for msg in request.chat_history[-5:]:  # keep last 5 messages only
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))

    # retrieve relevant docs
    docs = retriever.invoke(request.question)
    context = format_docs(docs)
    sources = get_sources(docs)

    # build chain with history
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "chat_history": history,
        "question": request.question
    })

    print(f"Sources: {sources}")
    print(f"Answer: {answer[:100]}...")

    return ChatResponse(answer=answer, sources=sources)
