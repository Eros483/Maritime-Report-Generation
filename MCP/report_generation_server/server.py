import llama_cpp
from datetime import datetime
from mcp.server.fastmcp import FastMCP
import json

mcp=FastMCP("Report Generation Usage")
model_path=f"C:\\Users\\caio\\code\\maritime_report_generation\\models\\dolphin3.0-llama3.2-3b-q5_k_m.gguf"

def load_model(path: str):
    '''
    Input: Model gguf Path
    Output: initialised llm
    '''
    global llm
    llm=llama_cpp.Llama(model_path=path, chat_format="llama-2", n_ctx=8192)
    return llm

#initialising everything
print("Initialising model.")
llm=load_model(model_path)

@mcp.tool(description="Generates report as per provided question, and previously queried data.")
async def report_generation(question, result):
    '''
    Generates reports.
    handles state["report"] and updates state["chat_history]
    '''
    llm.reset()
    print("Generating Report")
    question=question
    context=result
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

    try:
        response=llm.create_completion(
            prompt=prompt_template,
            temperature=temp,
            max_tokens=max_tokens
        )

        result=response['choices'][0]['text']
        result=result.replace("[/INST]", "").strip()
        llm.close()

        response={
            "question": question, 
            "data": result,
            "report": result, 
            "status": "Report Generated Succesfully"
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
