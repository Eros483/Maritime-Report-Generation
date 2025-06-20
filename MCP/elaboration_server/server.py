#MCP/elaboration_server/server.py

import llama_cpp
from mcp.server.fastmcp import FastMCP
from ..state_manager import read_state, update_field, add_chat_entry, get_chat_history_text 

mcp=FastMCP("Elaboration server")
model_path=f"C:\\Users\\caio\\code\\maritime_report_generation\\models\\dolphin3.0-llama3.2-3b-q5_k_m.gguf"

def load_model(path: str):
    '''
    Input: Model gguf Path
    Output: initialised llm
    '''
    global llm
    llm=llama_cpp.Llama(model_path=path, chat_format="llama-2", n_ctx=8192)
    return llm

print("Initialising model.")
llm=load_model(model_path)
print("Model ready for Analysis Generation.")

@mcp.tool(description="A tool that takes previously queried data from sql and a previously generated report, and provides a deeper analysis according to the user-requested input.")
def elaborate_on_response(input, report="", data="", history=""):
    '''
    elaborates on given information, and assigns it to input
    handles state["answer"] and updates state["chat_history"]
    '''
    try:
        llm.reset()
        print("Elaborating on given question")

        state=read_state()

        if not report:
            report = state.get("report", "")
        if not data:
            data = state.get("result", "")
        if not history:
            history = get_chat_history_text()
        
        question=input
        context=report
        
        update_field("query", question)

        elaboration_prompt=f"""
        <|im_start|>system
        Act as an experienced Indian military tactician.
        Explain it like someone who is a Indian naval commander.
        Answer the user's question using the given context and data in military parlance.
        Consider the user's previous queries carefully in the provided chat history.
        Enclose the responses with '''.

        context: {context}
        data: {data}
        chat history: {history}

        <|im_end|>
        <|im_start|>user
        Question: {question}
        <|im_end|>
        <|im_start|>assistant
        """
        temp=0.6
        max_tokens=250

        response=llm.create_completion(
            prompt=elaboration_prompt,
            temperature=temp,
            max_tokens=max_tokens
        )

        result=response['choices'][0]['text']
        result=result.replace("[/INST]", "").replace("'''", "").strip()

        update_field("elaboration", result)

        add_chat_entry(question, result, "Elaboration")

        print("Finished Response Elaboration")
        llm.reset()
        return result
    
    except Exception as e:
        error_msg = f"Error in elaboration: {str(e)}"
        print(error_msg)
        add_chat_entry(input, error_msg, "Elaboration (Error)")
        return error_msg

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')

