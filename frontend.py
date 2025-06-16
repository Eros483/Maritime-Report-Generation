import streamlit as st
from backend.main import load_model, app
from backend.tools import State

st.set_page_config(page_title="Report Generation and Chatbot", page_icon="⚓")
st.title("GenAI Report Assistant")
st.caption("Allows you to generate and view reports, while being able to chat with them as well")

#model path
model_path=f"C:\\Users\\caio\\code\\maritime_report_generation\\models\\dolphin3.0-llama3.2-3b-q5_k_m.gguf"

#initialising everything
llm=load_model(model_path)

if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

if "langgraph_state" not in st.session_state:
    langgraph_state = State(
        question="",
        query=None,
        result=None,
        report=None,
        answer=None,
        db=None,
        db_info=None,
        route=None,
        report_question=None,
        chat_history=[]
    )
    st.session_state.langgraph_state=langgraph_state

st.sidebar.subheader("Langgraph state inspector")
display_state = st.session_state.langgraph_state.copy()
if display_state.get("db"):
    display_state["db"] = f"SQLDatabase connected: {display_state['db'].dialect}" # Or any other useful string representation
else:
    display_state["db"] = "Not connected"

st.sidebar.json(display_state)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query=st.chat_input("Enter Query: ")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.langgraph_state["question"]=user_query
    with st.spinner("Processing Command..."):
        try:
            final_state=app.invoke(st.session_state.langgraph_state)
            assistant_response=final_state.get("answer", "No answer generated.")
            st.session_state.chat_history = final_state.get("chat_history", [])

            if st.session_state.chat_history:
                last_message = st.session_state.chat_history[-1]
                if last_message["role"] == "assistant":
                    with st.chat_message("assistant"):
                        st.markdown(last_message["content"])
                else:
                     with st.chat_message("assistant"):
                        st.markdown(assistant_response)
            st.session_state.langgraph_state = final_state

        except Exception as e:
            st.error(f"Error processing command: {e}")
            # Append error message to Streamlit's display history as well
            st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {e}"})
            with st.chat_message("assistant"):
                st.markdown(f"Error: {e}")