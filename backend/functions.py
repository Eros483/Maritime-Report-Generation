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
import markdown
import pdfkit
import tempfile

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

    route: str
    
    report_question: str
    chat_history: List[Dict[str, str]]

def load_model(path: str):
    '''
    Input: Model gguf Path
    Output: initialised llm
    '''
    print("---------------------------------\n\n")
    print("Loading model")
    print("---------------------------------\n\n")
    global llm
    llm=llama_cpp.Llama(model_path=path, chat_format="llama-2", n_ctx=8192)
    return llm

def assign_db(state: State)->State:
    '''
    handles state["db"] and state["db_info"]
    Assigns database to be used
    '''
    print("---------------------------------\n\n")
    print("Assigning Database")
    print("---------------------------------\n\n")
    sqlite_db=SQLDatabase.from_uri("sqlite:///sql_files/myDataBase.db")
    sqlite_info=sqlite_db.get_table_info()
    print("succesful assignment")
    print(sqlite_db)
    print(sqlite_info)
    state["db"]=sqlite_db
    state["db_info"]=sqlite_info
    print("---------------------------------\n\n")
    print(state["db"])
    print(state["db_info"])
    print("---------------------------------\n\n")
    print("left assigning database")
    return state

def router(state: State)->State:
    '''
    Decides what is to be done with user query, and assigns relevant route
    handles state["route"]
    '''
    print("Routing action to be taken")
    question=state["question"]
    llm.reset()
    print("Initialised answer")
    routing_prompt=f"""
    <|im_start|>system
    You are a general-purpose AI that helps people with questions.

    Given a question, your job is to categorize it into one of three categories:
    1. report: For when the user instructs, that instruct for report generation.
    2. analysis: For when the user requests for more analysis.
    3. general: For all other questions.
    Your response should be one word only.

    <|im_end|>
    <|im_start|>user
    Question: {question}
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
    print("---------------------------------\n\n")
    print(result)
    print("---------------------------------\n\n")
    state["route"]=result
    print("finished routing")
    llm.reset()
    return state

def write_sql_query(state: State)->State:
    llm.reset()
    dialect=state["db"].dialect
    top_k=5
    table_info=state["db_info"]
    input=state["question"]

    prompt_template=f"""
    <|im_start|>system

    ###TASK###
    Given an input question, create a syntactically correct {dialect} query to run to help find the answer. Unless the user specifies in his question a specific number of examples they wish to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.
    
    ###INSTRUCTIONS###:
    Never query for all the columns from a specific table, only ask for a the few relevant columns given the question. Generate only one query.
    Pay attention to use only the column names, and their values that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Prefer structured filters using equality.
    Only use the following tables:
    {table_info}

    ###SCHEMA DESCRIPTION###
    {table_info}

    ###USER FEEDBACK###
    You should avoid fuzzy pattern matching with LIKE '%chinese%' unless strictly necessary.
    You should not generate multiple queries.
    
    <|im_end|>
    <|im_start|>user
    Question: {input}
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
    state["query"]=result
    state["question"]=input
    llm.reset()
    print("---------------------------------\n\n")
    print(state["query"])
    print("---------------------------------\n\n")
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
    print("---------------------------------\n\n")
    print(state["result"])
    print("---------------------------------\n\n")
    print("finished executing query")
    return state

def report_generation(state: State)->State:
    '''
    Generates reports.
    handles state["report"] and updates state["chat_history]
    '''
    llm.reset()
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
    llm.reset()
    return state

def elaborate_on_response(state: State)->State:
    '''
    elaborates on given information, and assigns it to input
    handles state["answer"] and updates state["chat_history"]
    '''
    llm.reset()
    print("Elaborating on given question")
    question=state["question"]
    context=state["report"]
    data=state["result"]

    elaboration_prompt=f"""
    <|im_start|>system
    Act as an experienced Indian military tactician.
    Explain it like someone who is a Indian naval commander.
    Answer the user's {question} using the given {context} and {data} in military parlance.
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

def convert_report_to_pdf(state: State):
    report=state["report"]
    report_conversion_prompt=f"""
    <|im_start|>system
    Act as a report formatter, responsible for converting text into a proper format.
    The task is to format the user given {report} into the proper format as shown below.
    The output should be in the correct format as shown in the below examples.
    Ensure that the output adheres to the provided format.
    Make minimal changes to the actual information in the content, but encapsulate it in the below provided format.
    Make sure that the output is formatted correctly as provided.

    ###Format of Output
    From:  	Commanding Officer, Naval Support Activity Monterey
    To:     	Here B. Recipient, Organization
    Via:	(1) Here B. Intermediary, Organization (if needed for intermediary endorsement)
        (2) Number “via” recipients if more than 1; do not number if only 1

    Subj:  	LIMIT TO TWO LINES ALL CAPS NO ACRONYMS NO ABBREVIATIONS NO 
        PUNCTUATION (REPEAT SUBJECT LINE AT TOP OF SUBSEQUENT PAGES)

    Ref:    	(a) List as needed, or remove this line; must be referenced in the letter in order listed here
                (b) Include references/excerpts in routing package if they will inform the CO’s decision

    Encl:	(1) List as needed, or remove this line; number all enclosures here, even if just 1 
        (2) Must be referenced in the letter in order listed here

    1.  Left and right margins are always set at 1 inch.  Times New Roman 12 pitch font is preferred for Navy correspondence.  Single spacing between lines.  Double spacing between paragraphs/subparagraphs.  Send editable electronic copy to Admin for formatting/editing.

        a.	Indented ¼ inch.  

        b.	Indented ¼ inch.  If there is an a, there should be a b.

            (1) Indented ½ inch. 

            (2) Indented ½ inch.  If there is a (1), there should be a (2).

                (a) Indented ¾ inch.

                (b) Indented ¾ inch.  If there is an (a), there should be a (b).

                    1.  Indented 1 inch.

                    2.  Indented 1 inch.  If there is a 1, there should be a 2.

    2.  This is the second page of this letter.

    3.  For proper alignment, click on ruler across the top in Microsoft Word to set “soft” tab stops at ¼, ½, ¾, 1, 1 ¼ inches, etc.  Default tab stops set at 0.25” for each successive indentation.  Number pages 2 and up centered ½ inch from the bottom (including main letter and enclosures).

    4.  Do not use automatic formatting, bulleting, or “hard” stops that change page margins.  If copying from another document, select “keep text only” option to maintain proper formatting.

    5.  Break out acronyms on first use, then use the acronym the rest of the letter.
                                                                                                    


            I. M. COMMANDING

    Copy to:  (List here, as needed; keep to the minimum number necessary)
    Command Admin (N1/N04C)      Programs Integrator (N5)          CNRSW Chief of Staff              
    Operations (N3)                            Information Technology (N6)   Tenant Commands
    Public Works (N4)                       QOL Director (N9)                     NAVSUPPACT ANYWHERE

    ###Examples
    The correctly formatted output is enclosed in '''''.
    #Example 1:
    '''''
	12 Jun 25

    From:	Commanding Officer, Naval Support Activity Monterey  
    To:    	Here B. Recipient, Organization  
    Via:	(1) Here B. Intermediary, Organization  

    Subj:  	ASSESSMENT OF AIRCRAFT ACTIVITY WITHIN INDIAN TERRITORIAL WATERS

    Ref:	(a) Aircraft Movement Logs dated 12 Jun 25  
            (b) Surveillance Reports compiled by Indian Naval Aviation Command  

    Encl:	(1) Tabulated Movement Logs  
            (2) Visual Reconnaissance Summary  

    1.  This correspondence outlines the observed aircraft activity on 12 June 2025 within Indian territorial airspace, based on compiled data from naval surveillance systems and field reports. All contact was evaluated and determined to be of Friendly nature.  

        a.	Aircraft types observed were exclusively of Indian origin, with no detection of foreign or unidentified aerial assets.

        b.	Time-stamped reports indicated consistent aircraft movement in strategic sectors across Indian coastal boundaries, confirming a pattern of presence across key naval areas of responsibility (AOR).

            (1) Aerial maneuvers reported included advanced combat training routines:

                (a) Dogfighting simulations consistent with air-to-air engagement exercises.  

                (b) Split-S and Immelmann Turn maneuvers observed in multiple sectors.  

                (c) Barrel Rolls, Scissors maneuvers, and High-G turns performed—indicative of evasive and tactical training operations.

                (d) Zoom climbs observed, matching profiles typical of high-speed vertical ascent training.

            (2) All maneuvers occurred within authorized Indian airspace. Proximity to high-value installations remained within standard operational norms.

    2.  No hostile intent or unauthorized incursion was detected. All aircraft maintained expected communication and transponder compliance with the Indian Naval Aviation Command.  

    3.  The pattern of erratic but deliberate movement, high-speed turns, and advanced aerobatics aligns with known combat-readiness training routines or specialized reconnaissance tasks.  

    4.  Continued observation and logging are recommended to maintain situational awareness and flag any deviations from anticipated friendly patterns.

    5.  No escalation or response action is warranted at this time.

            I. M. COMMANDING

    Copy to:  
    Command Admin (N1/N04C)  
    Programs Integrator (N5)  
    CNRSW Chief of Staff  
    Operations (N3)  
    Information Technology (N6)  
    Tenant Commands  
    Public Works (N4)  
    QOL Director (N9)  
    NAVSUPPACT ANYWHERE

    '''''
    <|im_end|>
    <|im_start|>user
    Report: {report}
    <|im_end|>
    <|im_start|>assistant
    """
    temp=0.5
    max_tokens=1024

    response=llm.create_completion(
        prompt=report_conversion_prompt,
        temperature=temp,
        max_tokens=max_tokens
    )

    result=response['choices'][0]['text']
    result=result.replace("[/INST]", "").replace("'''", "").strip()
    
    return result

def pdf_result(state, file_path: str = "generated_report.pdf") -> bytes:
    formatted_report=convert_report_to_pdf(state)
    formatted_report=formatted_report.replace("*", "").replace("```", "")
    temp_html_file_path = None
    temp_pdf_file_path = None

    try:
        html_content = markdown.markdown(formatted_report)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp_html:
            tmp_html.write(html_content)
            temp_html_file_path = tmp_html.name
        print(f"Temporary HTML file created: {temp_html_file_path}")

        if file_path:
            # If user wants to save to a specific file, use that path
            output_pdf_actual_path = file_path
            return_bytes = False
        else:
                # If user wants bytes, create another temporary file for PDF output
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", mode="wb") as tmp_pdf:
                temp_pdf_file_path = tmp_pdf.name
            output_pdf_actual_path = temp_pdf_file_path
            return_bytes = True

        print(f"PDF output path for wkhtmltopdf: {output_pdf_actual_path}")
        
        pdfkit.from_file(temp_html_file_path, output_pdf_actual_path)
        print(f"PDF generated by pdfkit successfully at: {os.path.abspath(output_pdf_actual_path)}")

        if return_bytes:
            # Read the generated PDF file into bytes
            with open(output_pdf_actual_path, "rb") as f:
                pdf_bytes = f.read()
            return pdf_bytes
        else:
            # PDF was saved to disk, return None as per function signature
            return None

    except FileNotFoundError:
        # This specific error usually means wkhtmltopdf is not found
        error_msg = "Error: 'wkhtmltopdf' executable not found. Please ensure it's installed and in your system PATH."
        print(error_msg)
        raise RuntimeError(error_msg) from None # Re-raise for Streamlit to catch
    except Exception as e:
        print(f"An error occurred during PDF generation: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        raise # Re-raise the exception to be caught by Streamlit's try/except
    finally:
        # Clean up temporary files
        if temp_html_file_path and os.path.exists(temp_html_file_path):
            os.remove(temp_html_file_path)
            print(f"Cleaned up temporary HTML file: {temp_html_file_path}")
        if temp_pdf_file_path and os.path.exists(temp_pdf_file_path):
            os.remove(temp_pdf_file_path)
            print(f"Cleaned up temporary PDF file: {temp_pdf_file_path}")

def general_response(state: State):
    return