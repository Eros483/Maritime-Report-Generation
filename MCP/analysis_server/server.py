#MCP/analysis_server/server.py

import llama_cpp
from datetime import datetime
from mcp.server.fastmcp import FastMCP
import json
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
import re

from ..state_manager import read_state, update_field, add_chat_entry, get_chat_history_text

dialect="sqlite"
mcp=FastMCP("Analysis Generation")
model_path=f"C:\\Users\\caio\\code\\maritime_report_generation\\models\\dolphin3.0-llama3.2-3b-q5_k_m.gguf"

def assign_db():
    '''
    handles state["db"] and state["db_info"]
    Assigns database to be used
    '''
    sqlite_db=SQLDatabase.from_uri("sqlite:///../../sql_files/myDataBase.db")
    sqlite_info=sqlite_db.get_table_info()
    return sqlite_db, sqlite_info

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
print("Model ready for SQL communication and report Generation.")
db, db_info=assign_db()
print("Datbase Ready.")

def write_sql_query(question: str, db_info: str):
    llm.reset()
    top_k=5

    prompt_template=f"""
    <|im_start|>system

    ###TASK###
    Given an input question, create a syntactically correct {dialect} query to run to help find the answer. Unless the user specifies in his question a specific number of examples they wish to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.
    
    ###INSTRUCTIONS###:
    Never query for all the columns from a specific table, only ask for a the few relevant columns given the question. Generate only one query.
    Pay attention to use only the column names, and their values that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Prefer structured filters using equality.
    Only use the following tables:
    {db_info}

    ###SCHEMA DESCRIPTION###
    {db_info}

    ###USER FEEDBACK###
    You should avoid fuzzy pattern matching with LIKE '%chinese%' unless strictly necessary.
    You should not generate multiple queries.
    
    <|im_end|>
    <|im_start|>user
    Question: {question}
    <|im_end|>
    <|im_start|>assistant
    """
    temp=0.3
    max_tokens=300

    response=llm.create_completion(
        prompt=prompt_template,
        temperature=temp,
        max_tokens=max_tokens
    )

    result=response['choices'][0]['text']
    result=result.replace("[/INST]", "").replace("'''", "").replace("```", "").strip()
    result = re.sub(r'\b(submarines?|ships?|aircraft|helicopters?)\b',
           lambda m: {'submarine': 'subsurface', 'submarines': 'subsurface',
                      'ship': 'surface', 'ships': 'surface',
                      'aircraft': 'air', 'helicopter': 'air', 'helicopters': 'air'}[m.group(0).lower()],
           result, flags=re.IGNORECASE)
    result=result.lower()
    llm.reset()
    return result

def execute_query(query: str, db)->str:
    '''
    Runs SQL query on server and returns relevant tuples as a reponse.
    handles state["result"]
    '''
    print("Executing query")
    execute_query_tool=QuerySQLDataBaseTool(db=db)
    result=execute_query_tool.invoke(query)
    return result

def elaborate_on_response(input, data, history):
    '''
    elaborates on given information, and assigns it to input
    handles state["answer"] and updates state["chat_history"]
    '''
    llm.reset()
    print("Elaborating on given question")
    question=input
    data=data
    history=history

    elaboration_prompt=f"""
    <|im_start|>system
    Act as an experienced Indian military tactician.
    Explain it like someone who is a Indian naval commander.
    Answer the user's question using the given data in military parlance.
    Consider the user's previous queries carefully in the provided chat history.
    Enclose the responses with '''.

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

    print("Finished Response Elaboration")
    llm.reset()
    return result

@mcp.tool(description="A tool that takes natural language questions as input, generates relevant sql queries, executes them, and generates a succient analysis.")
def analysis(input, history=""):
    update_field("query", input)
    
    query=write_sql_query(input, db_info)
    update_field("sql_query", query)

    if not history:
        history=get_chat_history_text()

    result=execute_query(query, db)
    update_field("result", result)

    analysis=elaborate_on_response(input, result, history)
    update_field("analysis", analysis)

    add_chat_entry(input, analysis, "Analysis")
    
    return analysis

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')