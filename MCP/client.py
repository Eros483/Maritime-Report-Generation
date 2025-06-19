import asyncio
import os
from dotenv import load_dotenv
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatLlamaCpp
from mcp_use import MCPAgent, MCPClient
import traceback

model_path=f"C:\\Users\\caio\\code\\maritime_report_generation\\models\\dolphin3.0-llama3.2-3b-q5_k_m.gguf"
config_path=f"C:\\Users\\caio\\code\\maritime_report_generation\\MCP\\config.json"

async def main():
    try:
        #llm_agent=ChatLlamaCpp(model_path=model_path, temperature=0.3, max_tokens=4000, top_p=0.9, n_ctx=16384, n_batch=512, verbose=True, grammar_path=None)
        llm_agent=ChatOllama(model="llama3.2")

        print("\n\nConnecting to llm\n\n")
        client=MCPClient.from_config_file(filepath=config_path)

        print("\n\nTesting MCP server connection\n\n")
        tools=client.get_server_names()
        print(tools)

        if not tools:
            print("No tools found.")
            return

        agent=MCPAgent(llm=llm_agent, client=client, max_steps=30, verbose=True)

        query=input("Enter your question: ")

        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        try:
            response=await agent.run(query=query, max_steps=10)
            print(response)
        except Exception as e:
            print(f"Issue with query: {e}")

    except Exception as e:
        print(f"Failed initialisation. {e}")
        traceback.print_exc()

if __name__=="__main__":
    asyncio.run(main())