from langgraph.graph import StateGraph, END, START
import os
from .tools import State, load_model, assign_db, router, write_sql_query, execute_query, report_generation, elaborate_on_response, update_chat_history

workflow=StateGraph(State)

workflow.add_node("assign_db_node", assign_db)
workflow.add_node("router_node", router)
workflow.add_node("write_sql_node", write_sql_query)
workflow.add_node("execute_sql_node", execute_query)
workflow.add_node("report_gen_node", report_generation)
workflow.add_node("elaborate_node", elaborate_on_response)
workflow.add_node("update_chat_node", update_chat_history)

state: State={}

workflow.set_entry_point("assign_db_node")
workflow.add_edge("assign_db_node", "router_node")
workflow.add_conditional_edges(
    "router_node",
    lambda state: state["route"],
    {
        0: "write_sql_node",
        1: "elaborate_node",
    }
)
workflow.add_edge("write_sql_node", "execute_sql_node")
workflow.add_edge("execute_sql_node", "report_gen_node")

workflow.add_edge("report_gen_node", "update_chat_node")
workflow.add_edge("elaborate_node", "update_chat_node")

workflow.add_edge("update_chat_node", END)

app=workflow.compile()




