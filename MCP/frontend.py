import streamlit as st
import asyncio
import json
import traceback
from datetime import datetime
import sys
import os

# Add the MCP directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from langchain_ollama import ChatOllama
    from langchain_community.chat_models import ChatLlamaCpp
    from mcp_use import MCPAgent, MCPClient
    from state_manager import read_state, clear_state
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Configuration
MODEL_PATH = "C:\\Users\\caio\\code\\maritime_report_generation\\models\\dolphin3.0-llama3.2-3b-q5_k_m.gguf"
CONFIG_PATH = "C:\\Users\\caio\\code\\maritime_report_generation\\MCP\\config.json"

class ToolOutputExtractor:
    @staticmethod
    def extract_tool_result(agent_response: str) -> str:
        try:
            import re
            
            # Try to extract JSON first
            json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', agent_response, re.DOTALL)
            
            for json_match in json_matches:
                try:
                    parsed_json = json.loads(json_match)
                    if any(key in parsed_json for key in ['report', 'result', 'output', 'response']):
                        if 'report' in parsed_json:
                            return parsed_json['report']
                        elif 'result' in parsed_json:
                            return parsed_json['result']
                        elif 'output' in parsed_json:
                            return parsed_json['output']
                        elif 'response' in parsed_json:
                            return parsed_json['response']
                        else:
                            return json.dumps(parsed_json, indent=2)
                except json.JSONDecodeError:
                    continue
            
            # Try to extract tool result pattern
            tool_result_pattern = r'(?:Tool result:|Tool output:|Result:)\s*(.*?)(?:\n\n|$)'
            tool_match = re.search(tool_result_pattern, agent_response, re.DOTALL | re.IGNORECASE)
            if tool_match:
                return tool_match.group(1).strip()
            
            # Return substantial blocks
            lines = agent_response.split('\n')
            substantial_blocks = []
            current_block = []
            
            for line in lines:
                if line.strip():
                    current_block.append(line)
                else:
                    if current_block and len('\n'.join(current_block)) > 100:
                        substantial_blocks.append('\n'.join(current_block))
                    current_block = []
            
            if current_block and len('\n'.join(current_block)) > 100:
                substantial_blocks.append('\n'.join(current_block))
            
            if substantial_blocks:
                return substantial_blocks[-1]
            
            return agent_response
            
        except Exception as e:
            st.error(f"Error extracting tool result: {e}")
            return agent_response

class SmartToolAgent:
    def __init__(self, llm, client, max_steps=30, verbose=False):
        self.agent = MCPAgent(llm=llm, client=client, max_steps=max_steps, verbose=verbose)
        self.extractor = ToolOutputExtractor()

    async def run(self, query: str, max_steps: int = 10) -> str:
        enhanced_query = f"""
        IMPORTANT: You are a tool executor. Your job is to:
        1. Select the appropriate tool for this query.
        2. Execute the tool with the correct parameters.
        3. Return only the raw tool output without any additional commentary, analysis or interpretation.
        Query: {query}
        Do not add any summary, analysis or additional text. Just execute the tool and return its output directly.
        """
        try:
            full_response = await self.agent.run(query=enhanced_query, max_steps=max_steps)
            tool_output = self.extractor.extract_tool_result(full_response)
            return tool_output
        except Exception as e:
            return f"Error executing tool: {str(e)}"

@st.cache_resource
def initialize_agent():
    """Initialize the MCP agent with caching to avoid reloading"""
    try:
        # Use ChatOllama for faster initialization
        llm_agent = ChatOllama(model="llama3.2")
        
        # Initialize MCP client
        client = MCPClient.from_config_file(filepath=CONFIG_PATH)
        
        # Get available tools
        tools = client.get_server_names()
        
        if not tools:
            st.error("No tools found. Please check your MCP configuration.")
            return None, None, []
        
        agent = SmartToolAgent(llm=llm_agent, client=client, max_steps=30, verbose=True)
        
        return agent, client, tools
        
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        st.code(traceback.format_exc())
        return None, None, []

def main():
    st.set_page_config(
        page_title="Maritime Report Generation System",
        page_icon="🚢",
        layout="wide"
    )
    
    st.title("🚢 Maritime Report Generation System")
    st.markdown("---")
    
    # Initialize agent
    agent, client, tools = initialize_agent()
    
    if agent is None:
        st.stop()
    
    # Sidebar for tools and state
    with st.sidebar:
        st.header("🛠️ System Status")
        
        # Display available tools
        st.subheader("Available Tools")
        for tool in tools:
            st.write(f"✅ {tool}")
        
        st.markdown("---")
        
        # State management
        st.subheader("State Management")
        
        if st.button("🔄 Refresh State"):
            st.rerun()
        
        if st.button("🗑️ Clear State"):
            clear_state()
            st.success("State cleared!")
            st.rerun()
        
        # Display current state
        try:
            state = read_state()
            st.subheader("Current State")
            
            if state.get("last_updated"):
                st.write(f"**Last Updated:** {state['last_updated']}")
            
            if state.get("query"):
                st.write(f"**Last Query:** {state['query'][:50]}...")
            
            if state.get("chat_history"):
                st.write(f"**Chat History:** {len(state['chat_history'])} entries")
                
        except Exception as e:
            st.error(f"Error reading state: {e}")
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about maritime data, generate reports, or request analysis..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Processing your request..."):
                try:
                    # Run the agent
                    response = asyncio.run(agent.run(query=prompt, max_steps=10))
                    
                    # Display response
                    if response:
                        st.markdown(response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        error_msg = "No response generated. Please check your query and try again."
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        
                except Exception as e:
                    error_msg = f"Error processing request: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
                    # Show detailed error in expander
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
    
    # Quick action buttons
    st.markdown("---")
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Sample Report"):
            sample_query = "Generate a report on all naval activities in the last 24 hours"
            st.session_state.messages.append({"role": "user", "content": sample_query})
            st.rerun()
    
    with col2:
        if st.button("🔍 Analyze Current Data"):
            sample_query = "Analyze the current maritime situation and provide insights"
            st.session_state.messages.append({"role": "user", "content": sample_query})
            st.rerun()
    
    with col3:
        if st.button("📋 Get System Status"):
            sample_query = "What is the current status of all tracked vessels?"
            st.session_state.messages.append({"role": "user", "content": sample_query})
            st.rerun()
    
    # Example queries
    with st.expander("💡 Example Queries"):
        st.markdown("""
        **Report Generation:**
        - "Generate a report on all Chinese vessels in the last week"
        - "Create a summary of submarine activity near Indian borders"
        - "Report on all hostile contacts in the Arabian Sea"
        
        **Analysis:**
        - "Analyze the movement patterns of vessel ID 1001"
        - "What are the threats near Porbandar?"
        - "Identify any unusual naval activity"
        
        **Elaboration:**
        - "Elaborate on the previous report with more tactical details"
        - "Provide deeper analysis of the submarine movements"
        - "Explain the strategic implications of the current situation"
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            Maritime Report Generation System v1.0 | Built with Streamlit & MCP
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()