import asyncio
import os
from dotenv import load_dotenv
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatLlamaCpp
from mcp_use import MCPAgent, MCPClient
import traceback
import re
import json

model_path=f"C:\\Users\\caio\\code\\maritime_report_generation\\models\\dolphin3.0-llama3.2-3b-q5_k_m.gguf"
config_path=f"C:\\Users\\caio\\code\\maritime_report_generation\\MCP\\config.json"

class ToolOutputExtracter:
    @staticmethod
    def extract_tool_result(agent_response: str)->str:
        try:
            json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', agent_response, re.DOTALL)

            for json_match in json_matches:
                try:
                    parsed_json=json.loads(json_match)
                    if any(key in parsed_json for key in ['report', 'result', 'output', 'response']):
                        # Return the main content field
                        if 'report' in parsed_json:
                            return parsed_json['report']
                        elif 'result' in parsed_json:
                            return parsed_json['result']
                        elif 'output' in parsed_json:
                            return parsed_json['output']
                        elif 'response' in parsed_json:
                            return parsed_json['response']
                        else:
                            # Return the entire JSON if no specific field found
                            return json.dumps(parsed_json, indent=2)
                
                except json.JSONDecodeError:
                    continue
            
            tool_result_pattern = r'(?:Tool result:|Tool output:|Result:)\s*(.*?)(?:\n\n|$)'
            tool_match = re.search(tool_result_pattern, agent_response, re.DOTALL | re.IGNORECASE)
            if tool_match:
                return tool_match.group(1).strip()
            
            lines = agent_response.split('\n')
            substantial_blocks = []
            current_block = []
            
            for line in lines:
                if line.strip():
                    current_block.append(line)
                else:
                    if current_block and len('\n'.join(current_block)) > 100:  # Substantial content
                        substantial_blocks.append('\n'.join(current_block))
                    current_block = []
            
            if current_block and len('\n'.join(current_block)) > 100:
                substantial_blocks.append('\n'.join(current_block))
            
            if substantial_blocks:
                return substantial_blocks[-1]
            
            return agent_response
            
        except Exception as e:
            print(f"Error extracting tool result: {e}")
            return agent_response

class SmartToolAgent:
    def __init__(self, llm, client, max_steps=30, verbose=False):
        self.agent=MCPAgent(llm=llm, client=client, max_steps=max_steps, verbose=verbose)
        self.extractor=ToolOutputExtracter()

    async def run(self, query:str, max_steps: int=10)->str:
        enhanced_query=f"""
        IMPORTANT: You are a tool executor. Your job is to:
        1. Select the appropriate tool for this query.
        2. Execute the tool with the correct parameters.
        3. Return only the raw tool output without any additional commentary, analysis or interpretation.
        Query: {query}
        Do not add any summary, analysis or additional text. Just execute the tool and return its output directly.
        """
        try:
            full_response=await self.agent.run(query=enhanced_query, max_steps=max_steps)
            tool_output=self.extractor.extract_tool_result(full_response)
            return tool_output
        
        except Exception as e:
            return f"Error executing tool: {str(e)}"

async def main():
    try:
        #llm_agent=ChatLlamaCpp(model_path=model_path, temperature=0.3, max_tokens=4000, top_p=0.9, n_ctx=16384, n_batch=512, verbose=True, grammar_path=None)
        llm_agent=ChatOllama(model="llama3.2")

        print("\n\nConnecting to llm\n\n")
        client=MCPClient.from_config_file(filepath=config_path)

        print("\n\nTesting MCP server connection\n\n")
        tools=client.get_server_names()
        print(f"Available Tools: {tools}")

        if not tools:
            print("No tools found.")
            return

        agent=SmartToolAgent(llm=llm_agent, client=client, max_steps=30, verbose=True)

        query=input("Enter your question: ")

        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        try:
            response=await agent.run(query=query, max_steps=10)
            print("\n" + "="*60)
            print("Tool Output")
            print("="*60)
            print(response)

        except Exception as e:
            print(f"Issue with query: {e}")

    except Exception as e:
        print(f"Failed initialisation. {e}")
        traceback.print_exc()

if __name__=="__main__":
    asyncio.run(main())