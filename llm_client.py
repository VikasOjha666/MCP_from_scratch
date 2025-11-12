import asyncio
import json
from typing import List, Dict, Any, Optional,Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from utils import strip_internal_fields, sanitize_for_json, get_prompt, extract_call_from_text, call_mcp_sse,get_server_for_tool
from mcp_tools_ret_utils import index_tools_to_lancedb, fetch_top_k_tools_formatted
import re
import os
MCP_SERVER_URLS = os.environ.get('MCP_SERVER_URLS', '')
if MCP_SERVER_URLS:
    MCP_SERVER_URLS = MCP_SERVER_URLS.split(',')


MODEL_PATH = "gorilla-openfunctions-v2-GGUF/gorilla-openfunctions-v2-q4_K_M.gguf"
CALL_MARKER_KEY = "CALL_FUNCTION"

app = FastAPI(title="MCP")

# Globals populated at startup
server_map_dict: Dict[str, List[Dict[str, Any]]] = {}
llm: Optional[Llama] = None


class QueryRequest(BaseModel):
    query: str
    k: int = 2
    include_server_url: bool = True

def try_parse_function_call(text: str) -> Optional[Tuple[str, dict]]:
    """
    Try to extract a function call from model text. Returns (func_name, args_dict)
    or None if no plausible call detected.

    Handles patterns like:
      <<function>>func_name(arg=val)
      <<function>> func_name(arg=val)
      CALL_FUNCTION: func_name(arg=val)
      func_name(arg=val)   (last-resort)
    """
    if not text:
        return None
    txt = text.strip()

    # Common patterns to try (ordered)
    patterns = [
        r"<<function>>\s*([A-Za-z0-9_]+)\s*\((.*)\)",       # <<function>>name(args)
        r"CALL_FUNCTION\s*:?\s*([A-Za-z0-9_]+)\s*\((.*)\)", # CALL_FUNCTION: name(args)
        r"<<function>>(?:\s*)([A-Za-z0-9_]+)\s*\((.*)\)",   # extra safe
        r"^([A-Za-z0-9_]+)\s*\((.*)\)$",                    # name(args) (careful)
    ]

    for p in patterns:
        m = re.search(p, txt, flags=re.DOTALL)
        if not m:
            continue
        func_name = m.group(1)
        args_str = m.group(2).strip()
        args: dict = {}

        if args_str == "":
            return func_name, args

        # Split top-level commas (avoid splitting inside quotes)
        parts = [s.strip() for s in re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', args_str) if s.strip()]

        for part in parts:
            # handle key=value pairs or bare positional values
            if "=" in part:
                k, v = part.split("=", 1)
                k = k.strip()
                v = v.strip()
                # Try to literal_eval the value (numbers, lists, strings)
                try:
                    parsed_v = ast.literal_eval(v)
                except Exception:
                    # fallback: strip surrounding quotes, else keep raw string
                    parsed_v = v.strip('"').strip("'")
                args[k] = parsed_v
            else:
                # positional args — store under special key "_args"
                args.setdefault("_args", []).append(part)

        return func_name, args

    return None

async def discover_tools(server_urls: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Discover tools on each MCP server URL and return a mapping:
      { server_url: [ {name, description, parameters}, ... ], ... }
    This is adapted from your provided snippet; it is defensive about missing metadata.
    """
    server_map_dict_local: Dict[str, List[Dict[str, Any]]] = {}

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
        server_map_dict_local[server_url] = functions

    return server_map_dict_local


@app.on_event("startup")
async def startup_event():
    """Discover tools and index them at startup. Also create the Llama instance once."""
    global server_map_dict, llm

    # Discover MCP tools (async)
    try:
        server_map_dict = await discover_tools(MCP_SERVER_URLS)
        print("Discovered tools:", {k: len(v) for k, v in server_map_dict.items()})
    except Exception as e:
        print("Warning: discover_tools failed at startup:", e)

    # Index tools into lancedb (safe to call even if server_map_dict is empty)
    try:
        index_tools_to_lancedb(server_map_dict, db_path="./mcp_tools_lancedb", overwrite=True)
        print("Indexed tools to lancedb")
    except Exception as e:
        print("Warning: indexing to lancedb failed:", e)

    # Instantiate Llama in a threadpool to avoid blocking the event loop
    loop = asyncio.get_event_loop()

    def _create_llm():
        print("Creating Llama model (this may take a while)...")
        return Llama(model_path=MODEL_PATH, n_threads=8, n_ctx=2048, n_gpu_layers=35)

    try:
        llm = await loop.run_in_executor(None, _create_llm)
        print("Llama model created")
    except Exception as e:
        print("Error creating Llama model:", e)
        llm = None
    print(f"server_map_dict={server_map_dict}")

async def _call_llm(prompt: str, max_tokens: int = 2048) -> str:
    """Run the synchronous llm(prompt) in a threadpool and return the text string."""
    global llm
    if llm is None:
        raise RuntimeError("LLM not initialized")

    loop = asyncio.get_event_loop()

    def _sync_call():
        out = llm(prompt, max_tokens=max_tokens, echo=False)
        if isinstance(out, dict):
            return out.get("choices", [{}])[0].get("text") or out.get("text") or str(out)
        return str(out)

    return await loop.run_in_executor(None, _sync_call)


@app.post("/query")
async def run_query(req: QueryRequest):
    """Main endpoint. Accepts JSON: {"query":"Add two numbers 3 and 4.", "k":2}

    Flow:
      - fetch top-k tools from lancedb using fetch_top_k_tools_formatted
      - build prompt, call Llama
      - detect function call; if found, call the MCP tool via SSE and optionally follow up
      - return JSON with result and any tool output
    """
    try:
        # Step 1: retrieve candidate tools from lancedb
        functions = fetch_top_k_tools_formatted(req.query, k=req.k, include_server_url=req.include_server_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching tools from index: {e}")

    # Step 2: prepare prompt
    functions_no_internal = strip_internal_fields(functions, fields=("embedding",))
    safe_functions = sanitize_for_json(functions_no_internal)
    prompt = get_prompt(req.query, safe_functions)

    # Step 3: call LLM
    try:
        raw_text = await _call_llm(prompt, max_tokens=2048)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during LLM call: {e}")

    # Step 4: detect function call
    call = extract_call_from_text(raw_text, functions=functions)
    if call is None:
        # fallback parser for other formats like <<function>>name(arg=val)
        fallback = try_parse_function_call(raw_text)
        if fallback is None:
            # No function call: return model text as final answer
            return {
                "final_text": raw_text,
                "tool_called": None,
                "tool_result": None,
                "raw_model_output": raw_text,
            }
        else:
            func_name, func_args = fallback
    else:
        func_name, func_args = call

    # Step 5: execute MCP tool via SSE
    url_to_call=get_server_for_tool(func_name, server_map_dict)
    try:
        tool_result = await call_mcp_sse(func_name, func_args,url=url_to_call)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling MCP tool: {e}")

    # Step 6: Optional follow-up LLM call to compose final answer
    followup_prompt = (
        prompt
        + "\n\n(The tool call has been executed.)\n"
        + f"Tool name: {func_name}\nTool output: {tool_result}\n" +
        "Now the result need to be formatted in a human readable tone."
        + "Assistant:"
    )

    try:
        final_text = await _call_llm(followup_prompt, max_tokens=2048)
    except Exception as e:
        # If follow-up LLM fails, still return tool_result and raw_text
        print(f"Exception={str(e)}")
        return {
            "final_text": None,
            "tool_called": func_name,
            "tool_result": sanitize_for_json(tool_result),
            "raw_model_output": raw_text,
            "followup_error": str(e),
        }

    return {
        "final_text": final_text,
        "tool_called": func_name,
        "tool_result": sanitize_for_json(tool_result),
        "raw_model_output": raw_text,
    }


if __name__ == '__main__':
    import uvicorn

    # Run with: python fastapi_mcp_endpoint.py  OR: uvicorn fastapi_mcp_endpoint:app --reload
    uvicorn.run(app, host='0.0.0.0', port=3000, log_level='info')
