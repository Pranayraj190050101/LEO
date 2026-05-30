import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(
    page_title="LEO - Manulife Integration Assistant",
    page_icon=None,
    layout="centered"
)

st.title("LEO")
st.caption("Manulife Fieldglass to Workday Integration Support Assistant")
st.divider()

# initialise session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi! I am LEO, your Fieldglass to Workday integration support assistant. How can I help you today? You can ask me about integration errors, field mismatches, escalation contacts, or procurement policies.",
        "sources": []
    })

# display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")

# chat input
if question := st.chat_input("Ask LEO about your integration issue..."):

    # add user message
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "sources": []
    })
    with st.chat_message("user"):
        st.markdown(question)

    # get answer from backend
    with st.chat_message("assistant"):
        with st.spinner("LEO is thinking..."):
            try:
                # send full chat history for memory
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]  # exclude current question
                ]
                response = requests.post(
                    API_URL,
                    json={
                        "question": question,
                        "chat_history": history
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                else:
                    answer = "Sorry, I encountered an error. Please try again or contact the Integration Support Team via ServiceNow."
                    sources = []
            except Exception as e:
                answer = "Sorry, I could not connect to the backend. Please make sure the server is running."
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
