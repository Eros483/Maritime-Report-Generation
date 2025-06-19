from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
import llama_cpp
import re
from mcp.server.fastmcp import FastMCP
import json

dialect="sqlite"
mcp=FastMCP("SQLite Database Usage")
model_path=f"C:\\Users\\caio\\code\\maritime_report_generation\\models\\dolphin3.0-llama3.2-3b-q5_k_m.gguf"

def load_model(path: str):
    '''
    Input: Model gguf Path
    Output: initialised llm
    '''
    global llm
    llm=llama_cpp.Llama(model_path=path, chat_format="llama-2", n_ctx=8192)
    return llm

def assign_db():
    '''
    handles state["db"] and state["db_info"]
    Assigns database to be used
    '''
    sqlite_db=SQLDatabase.from_uri("sqlite:///../../sql_files/myDataBase.db")
    sqlite_info=sqlite_db.get_table_info()
    return sqlite_db, sqlite_info

print("Initialising model.")
llm=load_model(model_path)
print("Model ready.")
db, db_info=assign_db()
print("Datbase Ready.")

@mcp.tool(description="Generates sql query from natural language input.")
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
    llm.close()
    return result

@mcp.tool(description=f"Takes syntactically {dialect} correct queries, and executes it on sql database.")
def execute_query(query: str, db)->str:
    '''
    Runs SQL query on server and returns relevant tuples as a reponse.
    handles state["result"]
    '''
    print("Executing query")
    execute_query_tool=QuerySQLDataBaseTool(db=db)
    result=execute_query_tool.invoke(query)
    return result

@mcp.tool(description="A tool that takes natural language questions as input, and returns relevant data from a sql database.")
async def fetch_data_from_sql(question: str)->str:
    try:    
        print(f"Processing Question {question}")

        query=write_sql_query(question, db_info)
        print(f"Sql query written->\n\n {query}")

        result=execute_query(query, db)
        print(f"Executed query, received response->\n\n {result}")

        response={
            "question": question, 
            "query": query,
            "result": result, 
            "status": "sql query generated and executed succesfully."
        }

        return json.dumps(response, indent=2)


    except Exception as e:
        error_response={
            "question": question,
            "error": str(e),
            "status": "error"
        }
        print(f"Error processing request, {e}")
        return json.dumps(error_response, indent=2)


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')