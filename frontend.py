import streamlit as st
from backend.main import load_model, app
from backend.functions import State, pdf_result
import time

st.set_page_config(page_title="Report Generation and Chatbot", page_icon="⚓")
st.title("GenAI Report Assistant")
st.caption("Allows you to generate and view reports, while being able to chat with them as well")

#model path
model_path=f"C:\\Users\\caio\\code\\maritime_report_generation\\models\\dolphin3.0-llama3.2-3b-q5_k_m.gguf"

#initialising everything
llm=load_model(model_path)

if 'llm_backend_initialization' not in st.session_state:
    with st.spinner("Initializing backend"):
        pass
    st.session_state.llm_backend_initialization=True

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
        route="",
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

report_content=st.session_state.langgraph_state.get("report", "")

if report_content:
    if st.button("Download Report"):
        with st.spinner("Creating PDF document for report"):
            try:
                pdf_bytes=pdf_result(st.session_state.langgraph_state, file_path=None)

                if pdf_bytes:
                    st.download_button(
                        label="Click to download the PDF",
                        data=pdf_bytes,
                        file_name="report.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.error("No pdf bytes found, invalid report")

            except Exception as e:
                print(f"Probelm  in creating pdf report {e}")
                st.exception(e)
    else:
        st.info("Click button to download report.")

user_query=st.chat_input("Enter Query: ")

if user_query:
    start_time=time.time()
    previous_report_content=st.session_state.langgraph_state.get("report")
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.langgraph_state["question"]=user_query
    with st.spinner("Processing Command..."):
        try:
            final_state=app.invoke(st.session_state.langgraph_state)
            assistant_response=final_state.get("answer", "No answer generated.")
            st.session_state.chat_history = final_state.get("chat_history", [])

            end_time=time.time()
            response_time=end_time-start_time

            if st.session_state.chat_history:
                last_message = st.session_state.chat_history[-1]
                if last_message["role"] == "assistant":
                    with st.chat_message("assistant"):
                        st.markdown(last_message["content"])
                        st.caption(f"Response time: {response_time:.2f} seconds")
                        print(response_time)
                else:
                     with st.chat_message("assistant"):
                        st.markdown(assistant_response)
                        st.caption(f"Response time: {response_time:.2f} seconds")
                        print(response_time)
            st.session_state.langgraph_state = final_state

            new_report_content=st.session_state.langgraph_state.get("report")
            if new_report_content and new_report_content!=previous_report_content:
                st.toast("Report generated, refreshing page")
                st.rerun()
                print(response_time)

        except Exception as e:
            end_time=time.time()
            response_time=end_time-start_time
            st.error(f"Error processing command: {e}")
            # Append error message to Streamlit's display history as well
            st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {e}"})
            with st.chat_message("assistant"):
                st.markdown(f"Error: {e}")
                st.caption(f"Response time: {response_time:.2f} seconds")