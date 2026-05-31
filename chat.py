import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

FAISS_PATH = "data/faiss"
DOCS_PATH = "docs"

st.set_page_config(
    page_title="LEO - Manulife Integration Assistant",
    page_icon=None,
    layout="centered"
)

@st.cache_resource
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if not os.path.exists(FAISS_PATH):
        st.info("Building knowledge base for the first time, please wait...")
        documents = []
        for filename in os.listdir(DOCS_PATH):
            if filename.endswith(".pdf"):
                filepath = os.path.join(DOCS_PATH, filename)
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                documents.extend(docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        os.makedirs(FAISS_PATH, exist_ok=True)
        vectorstore.save_local(FAISS_PATH)
    else:
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

    return retriever, llm

def get_answer(question, chat_history, retriever, llm):
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

    docs = retriever.invoke(question)
    context = "\n\n".join([
        "[Source: " + doc.metadata.get("source", "unknown") + "]\n" + doc.page_content
        for doc in docs
    ])
    sources = list(set([doc.metadata.get("source", "unknown") for doc in docs]))

    history = []
    for msg in chat_history[-5:]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "chat_history": history,
        "question": question
    })

    return answer, sources

st.title("LEO")
st.caption("Manulife Fieldglass to Workday Integration Support Assistant")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi! I am LEO, your Fieldglass to Workday integration support assistant. How can I help you today? You can ask me about integration errors, field mismatches, escalation contacts, or procurement policies.",
        "sources": []
    })

retriever, llm = load_chain()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")

if question := st.chat_input("Ask LEO about your integration issue..."):
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "sources": []
    })
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("LEO is thinking..."):
            try:
                answer, sources = get_answer(
                    question,
                    st.session_state.messages[:-1],
                    retriever,
                    llm
                )
            except Exception as e:
                answer = "Sorry, I encountered an error. Please try again or contact the Integration Support Team via ServiceNow."
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    st.markdown(f"- {source}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

st.divider()
st.caption("LEO uses Manulife integration runbooks and procurement policies to answer your questions.")
