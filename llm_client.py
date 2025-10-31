import asyncio
import json
import re
import ast
from typing import List, Dict, Any
from llama_cpp import Llama
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
import aiohttp
from urllib.parse import urljoin
from utils import *
from mcp_tools_ret_utils import index_tools_to_lancedb, fetch_top_k_tools_formatted

MCP_SERVER_URLS = ["http://localhost:8000/sse"]
MODEL_PATH = "gorilla-openfunctions-v2-GGUF/gorilla-openfunctions-v2-q4_K_M.gguf"

CALL_MARKER_KEY = "CALL_FUNCTION"


async def discover_tools(server_urls: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Discover tools on each MCP server URL and return a mapping:
      { server_url: [ {name, description, parameters}, ... ], ... }
    """
    server_map_dict: Dict[str, List[Dict[str, Any]]] = {}

    for server_url in server_urls:
        functions: List[Dict[str, Any]] = []
        try:
            client = BasicMCPClient(server_url)
            tool_spec = McpToolSpec(client=client)

            # Wait for tool list asynchronously with a timeout
            tools = await asyncio.wait_for(tool_spec.to_tool_list_async(), timeout=10)

            for tool in tools:
                meta = getattr(tool, "_metadata", None)
                if meta is None:
                    # fallback: try to access tool metadata properties defensively
                    name = getattr(tool, "name", None) or getattr(tool, "fn", None) or "unknown"
                    description = getattr(tool, "description", "") or ""
                    params_raw = getattr(tool, "fn_schema_str", None)
                else:
                    name = getattr(meta, "name", None) or "unknown"
                    description = getattr(meta, "description", "") or ""
                    params_raw = getattr(meta, "fn_schema_str", None)

                # Try to parse parameters if it's a JSON string; otherwise keep as-is
                params = None
                try:
                    if isinstance(params_raw, str):
                        params = json.loads(params_raw)
                    else:
                        params = params_raw
                except Exception as e:
                    # If parsing fails, keep raw value (and log the error)
                    print(f"[discover_tools] Warning: failed to parse fn_schema_str for tool '{name}': {e}")
                    params = params_raw

                func_dict = {
                    "name": name,
                    "description": description,
                    "parameters": params,
                }
                functions.append(func_dict)

        except asyncio.TimeoutError:
            print(f"[discover_tools] Timeout while fetching tools from {server_url}")
        except Exception as e:
            print(f"[discover_tools] Error while contacting {server_url}: {e}")

        # Always set an entry (possibly empty) for this server_url
        server_map_dict[server_url] = functions

    return server_map_dict


async def main():
    # IMPORTANT: await the async discover_tools call
    server_map_dict = await discover_tools(MCP_SERVER_URLS)
    query="Add numbers 3 and 4 and give their sum."
    #functions=server_map_dict["http://localhost:8000/sse"]
    # index (run once, or whenever tools change)
    index_tools_to_lancedb(server_map_dict, db_path="./mcp_tools_lancedb", overwrite=True)

    # query -> get results in your requested schema
    query = "Add two numbers 3 and 4."
    functions = fetch_top_k_tools_formatted(query, k=2, include_server_url=True)
    functions_no_internal = strip_internal_fields(functions, fields=("embedding",))
    safe_functions = sanitize_for_json(functions_no_internal)
    prompt = get_prompt(query, safe_functions)
    print(f"prompt={prompt}")

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
