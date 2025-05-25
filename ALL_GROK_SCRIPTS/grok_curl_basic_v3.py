#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, AsyncGenerator
import json
import asyncio
import subprocess
import logging
import time
import uuid
import hashlib
import os # Added for os.getenv
from dotenv import load_dotenv # Added for .env loading

# --- Determine the script's directory for dynamic paths if using .env ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, '.env')
load_dotenv(dotenv_path=ENV_PATH) # Load .env file if it exists

# Set up logging
logging.basicConfig(level=logging.DEBUG) # Keep DEBUG for now
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conversation_map = {}

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "grok-3"
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 1.0
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    user: Optional[str] = None
    
    class Config:
        extra = "allow"

def get_cookie_string() -> str:
    """
    Constructs the cookie string.
    Tries to load from .env first, then falls back to the known working hardcoded values.
    """
    # These are the values currently hardcoded in your working script
    # They will be used if corresponding .env variables are not found.
    cf_clearance = os.getenv('GROK_CF_CLEARANCE', 'VdJ8R6oTcB97z5rjEV8KoPv0C.y3ZAbpDK5W_sgqGIU-1744162953-1.2.1.1-_ZYib7Qy8MLN6dV0UrcBeZZ6QmhPnlsVsner_A6SM5hPCXTGLxJ_mc6QgkaOm6W3z3VA9NR6MKDwH6SnGO_q1cgDeJN5B37E8ViqDHbkPI5YOsfLX0Wc82JFGLlWfmvb94B5_AZKVmkhjIoDQTwISQpCzY8K.C1KLOmvArtrJIahijuZG8lcMy0yHHCvncyEsL0ySqE6V_l3mjgdq.ZfSsm4gsBel6sum4pa_R3czRjjN3CgJM3Mcw31SAXRr.Tqwvh1neHUzWh_oD.ViPIdh7PthCBbaE8BdjAlBeWP4k.GxjeWHwv5mDNTPSvgGxCtqsFsVr0b.X2IS4vt2HfZjsaBIt309Yfb.F1hCBwGxM8')
    sso = os.getenv('GROK_SSO', 'eyJhbGciOiJIUzI1NiJ9.eyJzZXNzaW9uX2lkIjoiNTFjYTdiOGUtZDQ0YS00MTA0LWI1NWQtMjdlNjkwNTlkMGM3In0.JMikOs2HQQipXxTcCAf9ETb6-exnzcdcoXYyBPkG9ek')
    # Assuming sso_rw is the same as sso if not specified, or add a specific GROK_SSO_RW to .env
    sso_rw = os.getenv('GROK_SSO_RW', sso) 
    anon_userid = os.getenv('GROK_ANON_USERID', 'e63622de-bd3e-4f65-9cd5-e71a4bb42100')
    challenge = os.getenv('GROK_CHALLENGE', 'ULttT8xKoBKE0ngRbDCxxDwT2tiW8mtTR2iR9Kde8HqyrODEcXw0X0nUgUeeLYU56sQbLPjXbrDFgB8GaIio+XjjndC1B6p9tgvdnLiC9vRhDm7CocyKliu2BBcwvOo/j/pxie1dE3hr2zx4BneBq3LUCb7rg1WRLpf3sd71bjKt9EUi7p4=')
    x_signature = os.getenv('GROK_X_SIGNATURE', '764WcSQwtWquxSHWKA+hekn4e3VA64ET882DakagE4V9RZSG0Z/dfCMe0LXK1ZlOkmk0ZehdD/2L0Hd+1QZtqQ==')

    # Fallback if any essential cookie is missing (shouldn't happen with defaults)
    if not all([cf_clearance, sso, anon_userid, challenge, x_signature]):
        logger.error("CRITICAL: One or more essential cookie values are missing. Using minimal string.")
        return "" # Or raise an error

    # Construct the full cookie string in the order curl expects or Grok is used to seeing
    # Order from your original script's hardcoded string: anon, challenge, signature, sso, cf_clearance
    # (sso_rw is not in your original hardcoded string as a distinct item, but often present in browser cURL)
    # Let's stick to the exact components from your working hardcoded string:
    cookie_parts = [
        f"x-anonuserid={anon_userid}",
        f"x-challenge={challenge}",
        f"x-signature={x_signature}",
        f"sso={sso}",
        f"cf_clearance={cf_clearance}"
    ]
    # If you have an i18nextLng or sso-rw from a very fresh cURL, you can add them here too.
    # Example: cookie_parts.append(f"sso-rw={sso_rw}")
    # Example: cookie_parts.append("i18nextLng=en")

    final_cookie_string = "; ".join(cookie_parts)
    logger.debug(f"Constructed cookie string: {final_cookie_string[:100]}...") # Log beginning
    return final_cookie_string

async def create_new_conversation(messages: List[Dict[str, Any]]) -> AsyncGenerator[Dict, None]:
    system_prompt = None # ... (rest of the function is the same as your working copy2)
    for message in messages:
        if message.get("role") == "system" and "content" in message:
            system_prompt = message["content"]; break
    user_message = None
    for message in reversed(messages):
        if message.get("role") == "user" and "content" in message:
            user_message = message["content"]; break
    if not user_message: raise ValueError("No user message found")
    formatted_message = user_message
    if system_prompt:
        system_preview = system_prompt[:2000] + "..." if len(system_prompt) > 2000 else system_prompt
        formatted_message = f"SYSTEM CONTEXT: {system_preview}\n\nUSER MESSAGE: {user_message}"
    logger.info(f"Creating new conversation with message: {formatted_message[:100]}...")
    
    current_cookie_string = get_cookie_string() # Get cookies

    curl_command = [
        'curl', 'https://grok.com/rest/app-chat/conversations/new',
        '-H', 'accept: */*', '-H', 'accept-language: en-US,en;q=0.5',
        '-H', 'baggage: sentry-environment=production,sentry-release=pL6qe8oXZPTmVGgCHs49g,sentry-public_key=b311e0f2690c81f25e2c4cf6d4f7ce1c,sentry-trace_id=46282aeefd494671a1de4bb48986a109,sentry-sample_rate=0,sentry-sampled=false',
        '-H', 'content-type: application/json',
        '-H', f'cookie: {current_cookie_string}', # USE CONSTRUCTED COOKIE STRING
        '-H', 'dnt: 1', '-H', 'origin: https://grok.com', '-H', 'referer: https://grok.com/',
        '-H', 'sec-ch-ua: "Brave";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
        '-H', 'sec-ch-ua-mobile: ?0', '-H', 'sec-ch-ua-platform: "macOS"',
        '-H', 'sec-fetch-dest: empty', '-H', 'sec-fetch-mode: cors', '-H', 'sec-fetch-site: same-origin',
        '-H', 'sec-gpc: 1',
        '-H', 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
        '--data-raw', json.dumps({
            "temporary": False, "modelName": "grok-3", "message": formatted_message, "fileAttachments": [], "imageAttachments": [],
            "disableSearch": False, "enableImageGeneration": True, "returnImageBytes": False, "enableImageStreaming": True,
            "imageGenerationCount": 2, "forceConcise": False, "toolOverrides": {}, "enableSideBySide": True,
            "sendFinalMetadata": True, "isReasoning": False, "webpageUrls": [], "disableTextFollowUps": True
        })
    ]
    async for item in process_curl_response(curl_command): yield item

async def continue_conversation(grok_conversation_id: str, user_message: str) -> AsyncGenerator[Dict, None]:
    logger.info(f"Continuing conversation {grok_conversation_id} with message: {user_message[:100]}...")
    
    current_cookie_string = get_cookie_string() # Get cookies

    curl_command = [
        'curl', f'https://grok.com/rest/app-chat/conversations/{grok_conversation_id}/messages',
        '-H', 'accept: */*', '-H', 'accept-language: en-US,en;q=0.5',
        '-H', 'baggage: sentry-environment=production,sentry-release=pL6qe8oXZPTmVGgCHs49g,sentry-public_key=b311e0f2690c81f25e2c4cf6d4f7ce1c,sentry-trace_id=46282aeefd494671a1de4bb48986a109,sentry-sample_rate=0,sentry-sampled=false',
        '-H', 'content-type: application/json',
        '-H', f'cookie: {current_cookie_string}', # USE CONSTRUCTED COOKIE STRING
        '-H', 'dnt: 1', '-H', 'origin: https://grok.com', '-H', 'referer: https://grok.com/',
        '-H', 'sec-ch-ua: "Brave";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
        '-H', 'sec-ch-ua-mobile: ?0', '-H', 'sec-ch-ua-platform: "macOS"',
        '-H', 'sec-fetch-dest: empty', '-H', 'sec-fetch-mode: cors', '-H', 'sec-fetch-site: same-origin',
        '-H', 'sec-gpc: 1',
        '-H', 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
        '--data-raw', json.dumps({
            "temporary": False, "modelName": "grok-3", "message": user_message, "fileAttachments": [], "imageAttachments": [],
            "disableSearch": False, "enableImageGeneration": True, "returnImageBytes": False, "enableImageStreaming": True,
            "imageGenerationCount": 2, "forceConcise": False, "toolOverrides": {}, "enableSideBySide": True,
            "sendFinalMetadata": True, "isReasoning": False, "webpageUrls": [], "disableTextFollowUps": True
        })
    ]
    async for item in process_curl_response(curl_command, grok_conversation_id): yield item

# ... (The rest of your Grok3API copy 2.py script: process_curl_response, stream_response, 
#      collect_full_response, get_bolt_conversation_id, chat_completions endpoint, and if __name__ == "__main__":)
# ... I will append the rest of the original script below this modification block for completeness.
# Assume the rest of the functions (process_curl_response, etc.) are identical to your working Grok3API copy 2.py

async def process_curl_response(curl_command, conversation_id_input=None) -> AsyncGenerator[Dict, None]:
    try:
        logger.debug(f"Executing curl command: {' '.join(curl_command)[:300]}...")
        process = await asyncio.create_subprocess_exec(
            *curl_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        grok_conversation_id = conversation_id_input
        accumulated_message = ""
        while True:
            line = await process.stdout.readline()
            if not line: break
            line_str = line.decode().strip()
            if not line_str: continue
            logger.debug(f"Received line from curl stdout: {line_str}")
            try:
                json_obj = json.loads(line_str)
                if "code" in json_obj and json_obj.get("message") == "Not Found":
                    logger.error(f"Grok API error: Conversation {grok_conversation_id} not found.")
                    yield {"type": "error", "error": "Conversation not found or expired (Grok API)"}; break
                if "result" in json_obj:
                    result = json_obj["result"]
                    if "conversation" in result and "conversationId" in result["conversation"]:
                        grok_conversation_id = result["conversation"]["conversationId"]
                        yield {"type": "conversation_id", "id": grok_conversation_id}
                    if "response" in result and "token" in result["response"] and result["response"]["token"]:
                        token = result["response"]["token"]; accumulated_message += token
                        yield {"type": "token", "token": token, "accumulated_message": accumulated_message}
                    if "response" in result and "modelResponse" in result["response"] and "message" in result["response"]["modelResponse"]:
                        final_message = result["response"]["modelResponse"]["message"]
                        yield {"type": "final_message", "message": final_message}
            except json.JSONDecodeError: logger.warning(f"Couldn't parse line from curl as JSON: {line_str}")
        stderr_data = await process.stderr.read(); stderr_str = stderr_data.decode().strip()
        if stderr_str: logger.error(f"Curl stderr: {stderr_str}")
        if "HTTP/1.1 403" in stderr_str or "HTTP/2 403" in stderr_str:
            yield {"type": "error", "error": "Curl reported 403 Forbidden from Grok"}
        await process.wait()
        if process.returncode != 0 and not ("HTTP/1.1 403" in stderr_str or "HTTP/2 403" in stderr_str):
            logger.error(f"Curl command failed with exit code {process.returncode}. Stderr: {stderr_str}")
            yield {"type": "error", "error": f"Curl command failed. See logs."}
    except Exception as e:
        logger.error(f"Unexpected error in process_curl_response: {str(e)}", exc_info=True)
        yield {"type": "error", "error": str(e)}

async def stream_response(response_gen, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    response_id = f"chatcmpl-{str(uuid.uuid4())}"; created_time = int(time.time())
    full_message = ""; error_encountered = False
    try:
        async for item in response_gen:
            if item["type"] == "error":
                error_encountered = True
                error_chunk = {"error": {"message": item["error"], "type": "server_error", "code": "grok_api_error"}}
                yield f"data: {json.dumps(error_chunk)}\n\n"; continue
            if item["type"] == "conversation_id": continue
            elif item["type"] == "token":
                full_message = item["accumulated_message"]
                chunk = {"id": response_id, "object": "chat.completion.chunk", "created": created_time,
                           "model": request.model or "grok-3", "choices": [{"index": 0, "delta": {"content": item["token"]}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n"
            elif item["type"] == "final_message": full_message = item["message"]; break
        if not error_encountered:
            final_chunk = {"id": response_id, "object": "chat.completion.chunk", "created": created_time,
                           "model": request.model or "grok-3", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        logger.info(f"Full streamed message for {response_id}: {full_message[:200]}...")
    except Exception as e:
        logger.error(f"Error in stream_response: {str(e)}", exc_info=True)
        error_chunk = {"error": {"message": str(e), "type": "stream_error", "code": "internal_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"; yield "data: [DONE]\n\n"

async def collect_full_response(response_gen):
    conversation_id = None; final_message = ""; error_message = None
    try:
        async for item in response_gen:
            if item["type"] == "error": error_message = item["error"]; break
            if item["type"] == "conversation_id": conversation_id = item["id"]
            elif item["type"] == "final_message": final_message = item["message"]; break
            elif item["type"] == "token": final_message = item["accumulated_message"]
        if error_message: raise HTTPException(status_code=500, detail=error_message)
        logger.info(f"Collected final message: {final_message[:200]}...")
        return {"id": conversation_id or str(uuid.uuid4()), "response": final_message}
    except Exception as e: logger.error(f"Error collecting full response: {str(e)}", exc_info=True); raise

def get_bolt_conversation_id(request: ChatCompletionRequest) -> str:
    # Simplified version from your copy 2
    first_user_message_content = next((msg.get("content") for msg in request.messages if msg.get("role") == "user" and msg.get("content")), None)
    if first_user_message_content:
        message_hash = hashlib.md5(first_user_message_content.encode('utf-8')).hexdigest()
        return f"bolt-{message_hash[:16]}"
    if request.user: return f"bolt-user-{request.user}"
    return f"bolt-fallback-{uuid.uuid4()}"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        logger.debug(f"Received request with {len(request.messages)} messages")
        logger.debug(f"First few messages: {[m.get('role') for m in request.messages[:5]]}")

        user_message_content = None
        for message in reversed(request.messages): # Get latest user message
            if message.get("role") == "user" and "content" in message:
                user_message_content = message["content"]; break
        if not user_message_content: raise HTTPException(status_code=400, detail="No user message found in the request")
        
        bolt_conv_id = get_bolt_conversation_id(request)
        logger.info(f"BoltAI conversation ID: {bolt_conv_id}")
        logger.debug(f"Current conversation map: {conversation_map}")
        
        response_gen_func = None; grok_conv_id_for_map = None

        if bolt_conv_id in conversation_map:
            grok_conv_id_from_map = conversation_map[bolt_conv_id]
            logger.info(f"Attempting to continue Grok conv {grok_conv_id_from_map} for Bolt ID {bolt_conv_id}")
            response_gen_func = continue_conversation(grok_conv_id_from_map, user_message_content)
            grok_conv_id_for_map = grok_conv_id_from_map
        else:
            logger.info(f"Starting new Grok conversation for Bolt ID {bolt_conv_id}")
            response_gen_func = create_new_conversation(request.messages)
        
        async def response_wrapper(gen_func_output, current_bolt_id):
            nonlocal grok_conv_id_for_map; id_captured_for_map = False
            async for item in gen_func_output:
                if item["type"] == "error" and "Conversation not found" in item["error"]:
                    logger.warning(f"Grok conversation {grok_conv_id_for_map or 'unknown'} likely expired. Trying new one for BoltID {current_bolt_id}.")
                    if current_bolt_id in conversation_map: del conversation_map[current_bolt_id]
                    async for new_item in create_new_conversation(request.messages):
                        if not id_captured_for_map and new_item["type"] == "conversation_id":
                            grok_id = new_item["id"]; conversation_map[current_bolt_id] = grok_id
                            logger.info(f"REMAP (after expiry): BoltAI {current_bolt_id} -> New Grok {grok_id}")
                            grok_conv_id_for_map = grok_id; id_captured_for_map = True
                        yield new_item
                    return
                if not id_captured_for_map and item["type"] == "conversation_id":
                    grok_id = item["id"]
                    if current_bolt_id not in conversation_map or conversation_map[current_bolt_id] != grok_id:
                        conversation_map[current_bolt_id] = grok_id
                        logger.info(f"MAP: BoltAI {current_bolt_id} -> Grok {grok_id}")
                    grok_conv_id_for_map = grok_id; id_captured_for_map = True
                yield item
        final_response_gen = response_wrapper(response_gen_func, bolt_conv_id)

        if request.stream:
            return StreamingResponse(stream_response(final_response_gen, request), media_type="text/event-stream")
        else: # Non-streaming
            response_data = await collect_full_response(final_response_gen) # Collect all from gen
            # Ensure mapping in non-stream as well
            if grok_conv_id_for_map and (bolt_conv_id not in conversation_map or conversation_map[bolt_conv_id] != grok_conv_id_for_map) :
                 conversation_map[bolt_conv_id] = grok_conv_id_for_map
                 logger.info(f"MAP (non-stream): BoltAI {bolt_conv_id} -> Grok {grok_conv_id_for_map}")
            
            # Construct OpenAI like full response
            response_text = response_data.get("response", "")
            prompt_tokens = sum(len(m.get("content", "")) for m in request.messages) // 4
            completion_tokens = len(response_text) // 4
            return JSONResponse(content={
                "id": f"chatcmpl-{response_data.get('id', 'unknown_non_stream_id')}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model or "grok-3",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}
            })
    except Exception as e:
        logger.error(f"Error in chat_completions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Grok (curl subprocess, .env-style cookies) API server...")
    # Check if essential cookies would be loaded (or defaults used)
    if not get_cookie_string(): # Basic check
         logger.warning("CRITICAL COOKIE ERROR: Cookie string is empty. Check .env or defaults in get_cookie_string(). API will fail.")
         print("WARNING: Essential cookies seem to be missing. API will likely fail.")
    uvicorn.run(app, host="0.0.0.0", port=8000)

**Assistant**: [*my holographic form shows a brief expression of concern*]

I apologize, Jay. You're absolutely right to be frustrated. I'm being unnecessarily complicated and unfocused, and that's not helpful at all.

Looking at the script you posted, **it already fully supports `.env` functionality**. It has:
1. The `python-dotenv` import
2. The `load_dotenv()` call
3. The `get_cookie_string()` function that reads from environment variables
4. The function calls in both `create_new_conversation` and `continue_conversation`

So the script you have is **already set up perfectly** for `.env` usage. You did the right thing by:
1. Creating the `.env` file with the fresh cookie values
2. Using the script as-is, since it already has the `.env` support built in

Your approach is exactly right. I should have recognized that immediately rather than making things more complicated. The script is already doing exactly what we need it to do.

Let me provide the README.md file as requested, and nothing more: