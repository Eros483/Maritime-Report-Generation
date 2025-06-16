# GenAI Military Report Assistant

This project implements an AI-powered conversational agent designed to interact with a SQLite database to generate comprehensive military reports and answer related queries in natural language. It leverages a Large Language Model (LLM) for understanding user intent, generating SQL queries, summarizing data, and providing contextualized answers. The application is built with Streamlit for an interactive user interface and LangGraph for robust state management and workflow orchestration.

## Features

* **Natural Language to SQL:** Converts user questions into executable SQL queries against a SQLite database.
* **AI-Powered Report Generation:** Summarizes and contextualizes database query results into military-style reports using an LLM.
* **Conversational Interface:** Maintains chat history and allows users to ask follow-up questions or request elaborations on previous reports.
* **Intelligent Routing:** Automatically determines whether a new query requires database interaction (data retrieval) or an elaboration based on existing information.
* **Interactive User Interface:** A user-friendly chat interface built with Streamlit for seamless interaction.
* **PDF Report Export:** Ability to download generated reports as PDF files for offline viewing or archival.

## Technologies Used

* **Streamlit:** For building the interactive web-based user interface.
* **LangGraph:** Used for defining the conversational AI's state, nodes (functions), and conditional edges, enabling complex conversational flows and state persistence.
* **Llama.cpp (`llama-cpp-python`):** For efficient local inference with GGUF-formatted Large Language Models.
* **LangChain Community:** Provides utilities for interacting with SQL databases (`SQLDatabase`) and tools for SQL query execution (`QuerySQLDataBaseTool`).
* **SQLAlchemy:** Powers the database interaction layer via `SQLDatabase`.
* **SQLite:** The chosen database for storing and retrieving report data.
* **fpdf2:** A pure-Python library for generating PDF documents.
* **markdown-it-py:** A Markdown parser used to convert the LLM's Markdown report output into HTML for PDF generation.
* **Python `TypedDict`:** For defining a clear and structured schema for the LangGraph state.

## Project Structure
.
├── backend/
│   ├── main.py             # LangGraph application definition, loads LLM
│   ├── tools.py            # Defines the LangGraph State, nodes (functions) like assign_db, router, write_sql_query, etc.
│   └── init.py
├── models/
│   └── dolphin3.0-llama3.2-3b-q5_k_m.gguf # Placeholder for your downloaded LLM model
├── sql_files/
│   └── myDataBase.db       # Your SQLite database file
├── streamlit_app.py        # Streamlit frontend application
├── .gitignore              # Specifies files and directories to ignore in Git
└── README.md               # This file


## Setup

Follow these steps to set up and run the project locally.

### Prerequisites

* Python 3.9+
* Git (to clone the repository)

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name> # e.g., cd GenAIMilitaryReportAssistant
2. Create and Activate a Virtual Environment (Recommended)
Bash

python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
3. Install Dependencies
Bash

pip install streamlit langgraph langchain-community "llama-cpp-python[server]" fpdf2 markdown-it-py typing-extensions
4. Download the LLM Model
The project is configured to use a GGUF-formatted LLM. You need to download the model and place it in the models/ directory.

Model Used in Example: dolphin3.0-llama3.2-3b-q5_k_m.gguf

Download Location: You can find this and other GGUF models on Hugging Face. For example, search for "dolphin 3.0 llama 3.2 3b q5_k_m gguf" on Hugging Face.

Placement: Place the downloaded .gguf file into the models/ directory:
./models/dolphin3.0-llama3.2-3b-q5_k_m.gguf

Note: Large LLM files are often several gigabytes in size and are not included in the Git repository.

5. Database Setup
Ensure your SQLite database file (myDataBase.db) is located in the sql_files/ directory:

./sql_files/myDataBase.db

This database should contain the necessary tables and data that your LLM will query (e.g., report_data as seen in the prompt examples).

How to Run
Ensure your virtual environment is active.

Navigate to the project root directory (where streamlit_app.py is located) in your terminal.

Run the Streamlit application:

Bash

streamlit run streamlit_app.py
This command will open the Streamlit application in your default web browser.

Usage
Enter your query: Use the chat input field at the bottom to ask questions related to the data in your myDataBase.db.
Examples:
"Tell me what the Indian aircrafts are doing."
"Where are the US submarines?"
"Elaborate more on the selection of fruits available." (This would trigger the elaboration route)
View Report: If your query results in a report, it will be displayed in Markdown format above the chat area.
Download PDF: After a report is generated, a "Download Report as PDF" button will appear, allowing you to save the report to your local machine.
Continue Conversation: Ask follow-up questions to delve deeper into the current report or ask new questions to generate new reports.
State Inspector: The sidebar provides a real-time view of the LangGraph's internal state, useful for debugging and understanding the flow.