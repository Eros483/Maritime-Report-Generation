import os
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain import hub
import llama_cpp
from typing import TypedDict, Dict, List
from typing_extensions import Annotated
from langgraph.graph import StateGraph, START
from datetime import datetime
import re

llm=None
no_of_messages_retained=10

class State(TypedDict):
    question: str

    query: str

    result: str

    report: str
    answer: str

    db: SQLDatabase
    db_info: str

    route: int
    
    report_question: str
    chat_history: List[Dict[str, str]]

def load_model(path: str):
    '''
    Input: Model gguf Path
    Output: initialised llm
    '''
    print("Loading model")
    global llm
    llm=llama_cpp.Llama(model_path=path, chat_format="llama-2", n_ctx=8192)
    return llm

def assign_db(state: State)->State:
    '''
    handles state["db"] and state["db_info"]
    Assigns database to be used
    '''
    print("Assigning Database")
    sqlite_db=SQLDatabase.from_uri("sqlite:///sql_files/myDataBase.db")
    sqlite_info=sqlite_db.get_table_info()
    print("succesful assignment")
    print(sqlite_db)
    print(sqlite_info)
    state["db"]=sqlite_db
    state["db_info"]=sqlite_info
    print(state["db"])
    print(state["db_info"])
    print("left assigning database")
    return state

def router(state: State)->State:
    '''
    Decides what is to be done with user query, and assigns relevant route
    handles state["route"]
    '''
    print("Routing action to be taken")
    question=state["question"]
    answer=state["answer"]
    chat_history=state["chat_history"]
    print("Initialised answer")
    routing_prompt=f"""
    <|im_start|>system
    Act as a binary router for user questions, replying in 1 or 0.
    The Response is to be binary in nature, and only reply in 1 or 0.
    If chat_history is empty, or null, return 0. 
    else, The task is to decide if the {question} is a request for explaination or for futher information or if it is finding new data, or modifying existing data.
    Thus routing woud be of two kinds:
    1. An explaination request
        An explaination request, would imply the user asking for more information on a previous {answer}.

    2. A new data request
        A new data request, would imply the user is asking more data, not available in the previous {answer}. It would necessiate creating a new query.

    If it is an explaination request, return 1.
    If it is a new data request, return 0.

    The foutput should be enclosed inside '''.
    The format of the output is '''output'''.

    Structure your output such that, the first line consists of either 1 or 0.
    Any other information should only be from the second line onwards.

    If the user has explicitly asked for you to generate a report, return 0

    ###Examples
    Example question is enclosed inside ''''. Example output is enclosed inside '''.
    #Example1:
    ''''Generate a report on what the chinese are doing.''''
    '''0'''

    #Example2:
    ''''Generate a report''''
    '''0'''

    #Example3:
    ''''Elaborate more on the selection of fruits available.''''
    '''1'''

    #Example4:
    ''''No, This report does not contain sufficient information. I need to know more about the rafale. Regenerate the report accordingly.''''
    '''0'''

    #Example5:
    ''''Tell me more about the movement of the rafale.''''
    '''1'''

    #Example6:
    ''''Tell me about the submarine instead. Regenerate the report.''''
    '''0'''

    <|im_end|>
    <|im_start|>user
    Question: {input}
    <|im_end|>
    <|im_start|>assistant
    """
    temp=0.5
    max_tokens=100
    print("Creating response")
    response=llm.create_completion(
        prompt=routing_prompt,
        temperature=temp,
        max_tokens=max_tokens
    )
    print("Created Response")
    result=response['choices'][0]['text']
    result=result.replace("[/INST]", "").replace("'''", "").strip()
    print(result)
    match = re.search(r'[01]', result)
    digit = int(match.group(0)) if match else None
    print(digit)
    state["route"]=digit
    print("finished routing")
    return state


def write_sql_query(state: State)->State:
    '''
    Creates and assigns SQL query relevant to input
    handles state["query"]
    '''
    print("Writing sql query")
    dialect=state["db"].dialect
    top_k=5
    table_info=state["db_info"]
    input=state["question"]
    print(table_info)

    prompt_template=f"""
    <|im_start|>system
    Given an input question, create a syntactically correct {dialect} query to run to help find the answer. Unless the user specifies in his question a specific number of examples they wish to obtain, always ensure your query always has at least {top_k} results if available. You can order the results by a relevant column to return the most interesting examples in the database.
    Ensure the created query returns all available columns.
    Ensure that the columns queried exist in the {table_info}.
    Ensure that values queried exist in the {table_info}.
    Apply only user specified conditions.
    Ensure all criterion set by user has been utilised.
    Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Only use the following tables:
    {table_info}.
    The output should be enclosed inside '''.
    The output should contain only '''query'''.

    ###Additional Info
    Submarines are categorised as subsurface.
    Aircrafts and helicopters are categorised as air.
    Ships are categorised as surface.

    ###Examples
    Example question is enclosed inside ''''. Example output is enclosed inside '''.

    #Example1:
    ''''What are the names of the top 5 customers by revenue''''
    '''SELECT * FROM customers ORDER BY revenue DESC LIMIT 5;'''

    #Example2:
    ''''List the top 10 products sold in the last 6 months, including their names, categories, and total units sold, only for products that belong to the 'Electronics' category and have at least 100 units sold. Sort the results by total units sold in descending order.''''
    '''SELECT * FROM products p JOIN categories c ON p.category_id = c.id JOIN sales s ON p.id = s.product_id WHERE c.category_name = 'Electronics' AND s.sale_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH) GROUP BY p.name, c.category_name HAVING total_units_sold >= 100 ORDER BY total_units_sold DESC LIMIT 10;'''

    #Example3:
    ''''Tell me what the Indian aircrafts are doing''''
    '''SELECT * from report_data where category='air' and nationality='Indian' order by time desc limit 5'''

    #Example4:
    ''''Where are the US submarines?'''
    '''select * from report_data where category='subsurface' and nationality='US' order by time desc limit 5''

    <|im_end|>
    <|im_start|>user
    Question: {input}
    <|im_end|>
    <|im_start|>assistant
    """

    temp=0.5
    max_tokens=512

    temp=0.5
    max_tokens=512

    response=llm.create_completion(
        prompt=prompt_template,
        temperature=temp,
        max_tokens=max_tokens
    )

    result=response['choices'][0]['text']
    result=result.replace("[/INST]", "").replace("'''", "").strip()
    
    state["query"]=result
    print("Finished writing sql queries")
    return state

def execute_query(state: State):
    '''
    Runs SQL query on server and returns relevant tuples as a reponse.
    handles state["result"]
    '''
    print("Executing query")
    execute_query_tool=QuerySQLDataBaseTool(db=state["db"])
    result=execute_query_tool.invoke(state["query"])
    state["result"]=result
    print("finished executing query")
    return state

def report_generation(state: State)->State:
    '''
    Generates reports.
    handles state["report"] and updates state["chat_history]
    '''
    print("Generating Report")
    question=state["question"]
    context=state["result"]
    time=datetime.now().strftime("%d-%m-%y %H:%M:%S")
    prompt_template=f"""
    <|im_start|>system
    Act as an experienced Indian military tactician creating a report using {context}.
    Explain it like someone who is a Indian naval commander.
    Using the given {context}, answer the {question} in a precise manner using crisp military parlance.
    Ensure that the answer contains information from the provided {context}.
    The goal is to identify patterns in the {context} and relay necessary information.
    It should be a concise report, consisting of all the necessary information, highlighting patterns in data.

    You will be given a list of column labels, an explaination on what each label means, and then the {context}, which is a list of data tuples.
    Each element in the data tuple, corresponds to the column label at the same position.

    If the answer on the question is not in the provided context, tell the user, you can't answer the question on basis of the available data.
    Structure your response in markdown.
    Include Report Generation Time and date {time} in the report.

    #column labels
    id, name, latitude, longitude, range, bearing, course, speed, altitude, depth, reported_by, comment, hostility, category, nationality, location_wrt_naval_borders, closest_point_of_mil_interest, time, location

    #explaination of each label
    id: id is the unique identification number used for each target. Repeated occurences of the same id, mean information about the same target, and indicates a movement patter.
    name: Non-unique identification for each target. Useful for explaining in a more meaningful manner.
    latitude: latitudinal coordinate of each target. Non essential for the report.
    longitude: longitudinal coordinate of each target. Non essential for the report.
    range: Indicates how far the target is from the user.
    bearing: Indicates the direction of the target's position with respect to the user.
    course: Indicates the direction the target is moving in, with respect to True North.
    speed: Speed of the target.
    altitude: Height of the target.
    depth: Depth of the target.
    reported_by: Name of the entity which reported the target to the user.
    comment: Additional insights about the target.
    hostility: Indicates relationship between target and user. Important component to be considered in the report creation.
    category: Identifies nature of the target. Can be used to identify movement pattern as explained below. Essential to the report.
    nationality: Identifies origin country of the target.
    location_wrt_naval_borders: Identifies if target is inside Indian naval boundaries. If it is, it is important to be flagged in the report.
    closest_point_of_mil_interest: Identifies closest point of military interest to the target. Indicates prospective destination of the target. Might be used to indicate a pattern.
    time: Identifies time of capture of information of the target. Useful to identify patterns.
    location: Non essential for the report.

    Based on category and movement patterns,
    If the target is a ship, it's movement might be of the type
    Transit: Moving from point A to B (e.g. "underway transit through Strait of Hormuz").
    Patrol: Systematic movement within a designated area to monitor or deter.
    Station Keeping: Maintaining a relative position to other ships or a fixed point.
    Loitering: Holding in an area without aggressive movement; usually in anticipation.
    Maneuvering: Tactical repositioning (can include evasive, offensive, or formation-related changes).
    Shadowing: Following a foreign vessel at a set distance to monitor.

    If the target is a submarine, it's movement might be of the type
    Silent Running: No active sonar, minimal noise, passive listening only.
    Depth Excursion: Changing depth suddenly to avoid detection or torpedoes.
    Crazy Ivan: Sudden 180° turn at high speed to detect trailing enemies (Soviet/Russian tactic).
    Bottoming: Settling quietly on the sea floor to hide.
    Sprint-Drift: Burst of speed (sprint) followed by passive drift for listening.
    Evasion Patterns: Zig-zagging or random depth/speed changes to break sonar lock.
    Shadowing: Following a target vessel (esp. strategic like a carrier or SSBN).

    if the target is an aircraft, it's movement might of the type
    Dogfighting: Close-range air combat.
    Split-S: Rolling inverted and diving to reverse course.
    Immelmann Turn: Climbing half-loop followed by roll to reverse direction with altitude gain.
    Barrel Roll: Spiral roll to evade.
    Scissors: Close-range weaving maneuvers to force overshoot.
    High-G Turn: Tight turning using gravity to bleed enemy speed or positioning.
    Zoom Climb: Rapid vertical climb using momentum.

    
    ###Example:
    Question: Give me all information on the movement of the Rafale.
    Output:
    **OPERATIONAL REPORT: RAFALE ACTIVITY ANALYSIS (12 JUN 2025)**

    **TO:** Naval Command
    **TIME AND DATE OF GENERATION OF REPORT:** 12th June 2025 17:07:43

    **SUBJECT:** Assessment of Rafale Aircraft (ID 2001) Activity

    **SUMMARY:**
    Multiple reports by Arnab (ID 2001) indicate a friendly Indian Rafale aircraft conducting flights within Indian territorial waters near Porbandar. The aircraft exhibited erratic movements, varying speed, and fluctuating course over a brief observation period, suggesting potential training maneuvers or reconnaissance. All reported positions are well within Indian naval boundaries.

    **OBSERVATIONS & ANALYSIS:**

    1.  **Identity & Status:** Target ID 2001, identified as a Rafale aircraft of Indian nationality. Categorized as Friendly. This consistently indicates a known asset operating in our Area of Responsibility (AOR).
    2.  **Geographic & Temporal Span:** Data points range from 12:02:47 to 12:06:47 IST, all within Indian Waters, proximate to **Porbandar**. This signifies continued presence in a critical coastal region.
    3.  **Movement Patterns:**
        * **Course Volatility:** The aircraft's course shows significant variance (from 149.7 to 57.4 degrees True North). This, coupled with the observation comments strongly suggests non-linear flight paths consistent with training, evasive maneuvers, or complex patrol patterns rather than a direct transit.
        * **Speed Fluctuations:** Speed varied considerably, from a low of 385.0 knots to a high of 1300.8 knots. Such accelerations and decelerations are typical of air combat training or high-performance aerial reconnaissance.
        * **Range & Altitude:** Range decreased and then increased, while altitude remained constant at 120001 feet (nominal for high-altitude air operations, implying it's not conducting ground-level observation).
    4.  **Points of Interest:** The closest point of military interest remains consistently 'porbandar', reinforcing its operational focus in that vicinity.
    5.  **Hostility Assessment:** Confirmed Friendly status throughout all reports, mitigating immediate threat assessment.

    **CONCLUSION:**
    The Rafale (ID 2001) is engaging in what appears to be standard operational flights, likely training or specialized reconnaissance, within authorized Indian airspace. The Erratic movements and variable speeds are characteristic of advanced aerial exercises. Continued monitoring is advised to confirm operational intent and detect any deviation from expected friendly patterns.

    <|im_end|>
    <|im_start|>user
    Question: {question}
    <|im_end|>
    <|im_start|>assistant
    """

    temp=0.5
    max_tokens=1500

    response=llm.create_completion(
        prompt=prompt_template,
        temperature=temp,
        max_tokens=max_tokens
    )

    result=response['choices'][0]['text']
    result=result.replace("[/INST]", "").strip()
    
    state["report"]=result
    state["answer"]=result
    state["question"]=question
    state["report_question"]=question
    print("Finished report generation")
    return state

def elaborate_on_response(state: State)->State:
    '''
    elaborates on given information, and assigns it to input
    handles state["answer"] and updates state["chat_history"]
    '''
    print("Elaborating on given question")
    question=state["question"]
    context=state["report"]
    data=state["result"]
    chat_history=state["chat_history"]

    elaboration_prompt=f"""
    <|im_start|>system
    Act as an experienced Indian military tactician.
    Explain it like someone who is a Indian naval commander.
    Answer the user's {question} using the given {context} and {data}.
    Consider the previous questions and responses the user had received, {chat_history}, while creating the output to ensure there is no redundancy while creating response.
    Be precise and concise in nature, with short, to the point responses in military parlance.
    Enclose the responses with '''.
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

    state["answer"]=result
    print("Finished Response Elaboration")
    llm.reset()
    return state

def update_chat_history(state: State)->State:
    '''
    handles state["chat_history"]
    '''
    print("Entered Updating Chat history")
    question=state["question"]
    answer=state["answer"]

    chat_history=state.get("chat_history", [])
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})

    if len(chat_history)>no_of_messages_retained:
        chat_history=chat_history[-no_of_messages_retained]

    state["chat_history"]=chat_history
    print("Left Updating Chat history")
    return state