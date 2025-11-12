import asyncio
import json
import re
import ast
from llama_cpp import Llama
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
import aiohttp
from urllib.parse import urljoin
from utils import *

MCP_SERVER_URL = "http://localhost:8000/sse"
MODEL_PATH = "gorilla-openfunctions-v2-GGUF/gorilla-openfunctions-v2-q4_K_M.gguf"

CALL_MARKER_KEY = "CALL_FUNCTION"



async def main():
    client = BasicMCPClient(MCP_SERVER_URL)
    tool_spec = McpToolSpec(client=client)

    # Wait for tool list asynchronously
    tools = await asyncio.wait_for(tool_spec.to_tool_list_async(), timeout=10)

    functions = []
    for tool in tools:
        meta = tool._metadata
        try:
            params = meta.fn_schema_str
            if isinstance(params, str):
                params = json.loads(params)
        except Exception as e:
            print(f"Error parsing fn_schema_str for tool {meta.name}: {e}")
            continue

        func_dict = {
            "name": meta.name,
            "description": meta.description,
            "parameters": params
        }
        functions.append(func_dict)

    query = "Convert 32 celcius to fahrenheit."
    prompt = get_prompt(query, functions)
    print(f"Prompt={prompt}")

    llm = Llama(model_path=MODEL_PATH, n_threads=8, n_ctx=2048, n_gpu_layers=35)

    out = llm(prompt, max_tokens=2048, echo=False)

    print(f"Out={out}")

    if isinstance(out, dict):
        text = out.get("choices", [{}])[0].get("text") or out.get("text") or str(out)
    else:
        text = str(out)

    print("Model decision / raw output:\n", text)

    call = extract_call_from_text(text, functions=functions)
    if call is None:
        print("\nNo function call detected — returning model output as final response.")
        return

    func_name, func_args = call
    print(f"\nDetected function call: {func_name} with args: {func_args}")

    try:
        tool_result = await call_mcp_sse(func_name, func_args)
    except Exception as e:
        print("Error while calling MCP SSE:", e)
        return

    print("\nTool (MCP) result:\n", tool_result)

    # OPTIONAL: feed tool_result back into the model for a final answer
    followup_prompt = (
        prompt
        + "\n\n(The tool call has been executed.)\n"
        + f"Tool name: {func_name}\nTool output: {tool_result}\n"
        + "Assistant:"
    )
    print(f"followup_prompt={followup_prompt}")
    followup_out = llm(followup_prompt, max_tokens=1024, echo=False)
    if isinstance(followup_out, dict):
        final_text = followup_out.get("choices", [{}])[0].get("text") or followup_out.get("text") or str(followup_out)
    else:
        final_text = str(followup_out)

    print("\nFinal assistant answer incorporating tool output:\n", final_text)


if __name__ == "__main__":
    asyncio.run(main())
