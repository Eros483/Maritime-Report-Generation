#MCP/state_manager.py

import json
import os
from datetime import datetime

STATE_FILE="mcp_state.json"

def write_state(state):
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing state: {e}")

def read_state():
    if not os.path.exists(STATE_FILE):
        default_state={
            "query": "",
            "sql_query": "",
            "result": "",
            "report": "",
            "analysis": "",
            "elaboration": "",
            "chat_history": [],
            "last_updated": ""
        }
        write_state(default_state)
        return default_state

    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {
            "query": "",
            "sql_query": "",
            "result": "",
            "report": "",
            "analysis": "",
            "elaboration": "",
            "chat_history": [],
            "last_updated": ""
        }

def update_field(field_name, value):
    state=read_state()
    state[field_name]=value
    state["last_updated"]=datetime.now().isoformat()
    write_state(state)

def add_chat_entry(user_input, response, tool_used):
    state=read_state()

    new_entry={
        "timestamp":datetime.now().isoformat(),
        "user_input":user_input,
        "response":response,
        "tool_used":tool_used
    }

    state["chat_history"].append(new_entry)

    if len(state["chat_history"])>5:
        state["chat_history"]=state["chat_history"][-5:]

    state["last_updated"]=datetime.now().isoformat()
    write_state(state)

def get_chat_history_text():
    state=read_state()
    history_text=""
    for entry in state["chat_history"]:
        history_text += f"User: {entry['user_input']}\n"
        history_text += f"Assistant ({entry['tool_used']}): {entry['response']}\n"
        history_text += f"Time: {entry['timestamp']}\n\n"
    return history_text

def clear_state():
    default_state={
        "query": "",
        "sql_query": "",
        "result": "",
        "report": "",
        "analysis": "",
        "elaboration": "",
        "chat_history": [],
        "last_updated": datetime.now().isoformat()
    }
    write_state(default_state)