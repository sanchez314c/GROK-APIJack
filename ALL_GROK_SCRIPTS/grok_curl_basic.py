#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Generator, AsyncGenerator
# import httpx # Not used in this version for Grok calls
import json
import asyncio
import subprocess # Used for curl
import logging
import time
import uuid
import hashlib

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory conversation store to track which BoltAI conversations map to which Grok conversations
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

async def create_new_conversation(messages: List[Dict[str, Any]]) -> AsyncGenerator[Dict, None]:
    """Create a new conversation in Grok"""
    
    system_prompt = None
    for message in messages:
        if message.get("role") == "system" and "content" in message:
            system_prompt = message["content"]
            break
    
    user_message = None
    for message in reversed(messages):
        if message.get("role") == "user" and "content" in message:
            user_message = message["content"]
            break
    
    if not user_message:
        raise ValueError("No user message found")
    
    formatted_message = user_message
    if system_prompt:
        system_preview = system_prompt[:2000] + "..." if len(system_prompt) > 2000 else system_prompt
        formatted_message = f"SYSTEM CONTEXT: {system_preview}\n\nUSER MESSAGE: {user_message}"
    
    logger.info(f"Creating new conversation with message: {formatted_message[:100]}...")
    
    # <<< CORTANA: PASTE YOUR FULL, FRESH COOKIE STRING HERE (after 'cookie: ') >>>
    # Example: 'cookie: cf_clearance=VALUE1; sso=VALUE2; x-anonuserid=VALUE3; ... etc ...'
    # Get this from your browser's "Copy as cURL" for a successful Grok request, from the -b or --cookie part.
    fresh_cookie_string = 'YOUR_FRESH_AND_COMPLETE_COOKIE_STRING_HERE' 

    curl_command = [
        'curl',
        'https://grok.com/rest/app-chat/conversations/new',
        '-H', 'accept: */*',
        '-H', 'accept-language: en-US,en;q=0.5',
        '-H', 'baggage: sentry-environment=production,sentry-release=pL6qe8oXZPTmVGgCHs49g,sentry-public_key=b311e0f2690c81f25e2c4cf6d4f7ce1c,sentry-trace_id=46282aeefd494671a1de4bb48986a109,sentry-sample_rate=0,sentry-sampled=false', # This baggage might become stale
        '-H', 'content-type: application/json',
        '-H', f'cookie: {fresh_cookie_string}', # Cookie string is now dynamic via variable
        '-H', 'dnt: 1',
        '-H', 'origin: https://grok.com',
        '-H', 'referer: https://grok.com/',
        '-H', 'sec-ch-ua: "Brave";v="135", "Not-A.Brand";v="8", "Chromium";v="135"', # These sec-ch-ua headers might need updating to match your browser
        '-H', 'sec-ch-ua-mobile: ?0',
        '-H', 'sec-ch-ua-platform: "macOS"',
        '-H', 'sec-fetch-dest: empty',
        '-H', 'sec-fetch-mode: cors',
        '-H', 'sec-fetch-site: same-origin',
        '-H', 'sec-gpc: 1',
        '-H', 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36', # User agent might need updating
        '--data-raw', json.dumps({
            "temporary": False, "modelName": "grok-3", "message": formatted_message,
            "fileAttachments": [], "imageAttachments": [], "disableSearch": False,
            "enableImageGeneration": True, "returnImageBytes": False, "enableImageStreaming": True,
            "imageGenerationCount": 2, "forceConcise": False, "toolOverrides": {},
            "enableSideBySide": True, "sendFinalMetadata": True, "isReasoning": False,
            "webpageUrls": [], "disableTextFollowUps": True
        })
    ]
    
    async for item in process_curl_response(curl_command):
        yield item

async def continue_conversation(grok_conversation_id: str, user_message: str) -> AsyncGenerator[Dict, None]:
    """Continue an existing Grok conversation"""
    
    logger.info(f"Continuing conversation {grok_conversation_id} with message: {user_message[:100]}...")
    
    # <<< CORTANA: PASTE YOUR FULL, FRESH COOKIE STRING HERE (after 'cookie: ') >>>
    # Example: 'cookie: cf_clearance=VALUE1; sso=VALUE2; x-anonuserid=VALUE3; ... etc ...'
    # Use the SAME fresh cookie string as in create_new_conversation.
    fresh_cookie_string = 'YOUR_FRESH_AND_COMPLETE_COOKIE_STRING_HERE'

    curl_command = [
        'curl',
        f'https://grok.com/rest/app-chat/conversations/{grok_conversation_id}/messages',
        '-H', 'accept: */*',
        '-H', 'accept-language: en-US,en;q=0.5',
        '-H', 'baggage: sentry-environment=production,sentry-release=pL6qe8oXZPTmVGgCHs49g,sentry-public_key=b311e0f2690c81f25e2c4cf6d4f7ce1c,sentry-trace_id=46282aeefd494671a1de4bb48986a109,sentry-sample_rate=0,sentry-sampled=false', # This baggage might become stale
        '-H', 'content-type: application/json',
        '-H', f'cookie: {fresh_cookie_string}', # Cookie string is now dynamic via variable
        '-H', 'dnt: 1',
        '-H', 'origin: https://grok.com',
        '-H', 'referer: https://grok.com/',
        '-H', 'sec-ch-ua: "Brave";v="135", "Not-A.Brand";v="8", "Chromium";v="135"', # These sec-ch-ua headers might need updating
        '-H', 'sec-ch-ua-mobile: ?0',
        '-H', 'sec-ch-ua-platform: "macOS"',
        '-H', 'sec-fetch-dest: empty',
        '-H', 'sec-fetch-mode: cors',
        '-H', 'sec-fetch-site: same-origin',
        '-H', 'sec-gpc: 1',
        '-H', 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36', # User agent might need updating
        '--data-raw', json.dumps({
            "temporary": False, "modelName": "grok-3", "message": user_message,
            "fileAttachments": [], "imageAttachments": [], "disableSearch": False,
            "enableImageGeneration": True, "returnImageBytes": False, "enableImageStreaming": True,
            "imageGenerationCount": 2, "forceConcise": False, "toolOverrides": {},
            "enableSideBySide": True, "sendFinalMetadata": True, "isReasoning": False,
            "webpageUrls": [], "disableTextFollowUps": True
        })
    ]
    
    async for item in process_curl_response(curl_command, grok_conversation_id):
        yield item

async def process_curl_response(curl_command, conversation_id_input=None) -> AsyncGenerator[Dict, None]:
    """Process the curl response and yield tokens and messages"""
    try:
        logger.debug(f"Executing curl command: {' '.join(curl_command)[:300]}...") # Log more of the command
        process = await asyncio.create_subprocess_exec(
            *curl_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        grok_conversation_id = conversation_id_input
        accumulated_message = ""
        
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            line_str = line.decode().strip()
            if not line_str: continue
            logger.debug(f"Received line from curl stdout: {line_str}")
            try:
                json_obj = json.loads(line_str)
                if "code" in json_obj and json_obj.get("message") == "Not Found": # Error from Grok API
                    logger.error(f"Grok API error: Conversation {grok_conversation_id} not found.")
                    yield {"type": "error", "error": "Conversation not found or expired (Grok API)"}
                    break
                if "result" in json_obj:
                    result = json_obj["result"]
                    if "conversation" in result and "conversationId" in result["conversation"]:
                        grok_conversation_id = result["conversation"]["conversationId"]
                        yield {"type": "conversation_id", "id": grok_conversation_id}
                    if "response" in result and "token" in result["response"] and result["response"]["token"]:
                        token = result["response"]["token"]
                        accumulated_message += token
                        yield {"type": "token", "token": token, "accumulated_message": accumulated_message}
                    if "response" in result and "modelResponse" in result["response"] and "message" in result["response"]["modelResponse"]:
                        final_message = result["response"]["modelResponse"]["message"]
                        yield {"type": "final_message", "message": final_message}
                        # No return here, let stdout fully drain
            except json.JSONDecodeError:
                logger.warning(f"Couldn't parse line from curl as JSON: {line_str}")
                continue
        
        stderr_data = await process.stderr.read()
        stderr_str = stderr_data.decode().strip()
        if stderr_str: # Log any output from curl's stderr
            logger.error(f"Curl stderr: {stderr_str}")
            # Check for common curl errors indicating a 403 or similar
            if "HTTP/1.1 403" in stderr_str or "HTTP/2 403" in stderr_str:
                 yield {"type": "error", "error": "Curl reported 403 Forbidden from Grok"}


        await process.wait()
        if process.returncode != 0:
            logger.error(f"Curl command failed with exit code {process.returncode}. Stderr: {stderr_str}")
            # Don't yield generic error if we already yielded a specific one like 403
            if not ("HTTP/1.1 403" in stderr_str or "HTTP/2 403" in stderr_str):
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
            elif item["type"] == "final_message": full_message = item["message"]; break # Break after final to ensure clean exit
        
        if not error_encountered:
            final_chunk = {"id": response_id, "object": "chat.completion.chunk", "created": created_time,
                           "model": request.model or "grok-3", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n" # Always send DONE
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
            elif item["type"] == "token": final_message = item["accumulated_message"] # Keep updating if no final_message event comes
        if error_message: raise HTTPException(status_code=500, detail=error_message)
        logger.info(f"Collected final message: {final_message[:200]}...")
        return {"id": conversation_id or str(uuid.uuid4()), "response": final_message}
    except Exception as e: logger.error(f"Error collecting full response: {str(e)}", exc_info=True); raise

def get_bolt_conversation_id(request: ChatCompletionRequest) -> str:
    user_msgs_content = "".join(msg.get("content","")[:200] for msg in request.messages if msg.get("role")=="user" and "content" in msg and not msg["content"].lower().startswith(("system prompt:","generate a concise","user:","system:")))
    if user_msgs_content: return f"bolt-{hashlib.md5(user_msgs_content.encode('utf-8')).hexdigest()[:16]}"
    if request.user: return f"bolt-user-{request.user}"
    return f"bolt-fallback-{uuid.uuid4()}"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        user_message_content = None
        for message in reversed(request.messages):
            if message.get("role") == "user" and "content" in message:
                user_message_content = message["content"]; break
        if not user_message_content: raise HTTPException(status_code=400, detail="No user message content found")
        
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
            response_gen_func = create_new_conversation(request.messages) # Pass full messages for new conv context
        
        async def response_wrapper(gen_func_output, current_bolt_id):
            nonlocal grok_conv_id_for_map
            id_captured_for_map = False
            async for item in gen_func_output:
                if item["type"] == "error" and "Conversation not found" in item["error"]: # Handle expired Grok conversation
                    logger.warning(f"Grok conversation {grok_conv_id_for_map or 'unknown'} likely expired. Trying to start a new one for BoltID {current_bolt_id}.")
                    if current_bolt_id in conversation_map: del conversation_map[current_bolt_id] # Clear stale mapping
                    async for new_item in create_new_conversation(request.messages): # Create new
                        if not id_captured_for_map and new_item["type"] == "conversation_id":
                            grok_id = new_item["id"]; conversation_map[current_bolt_id] = grok_id
                            logger.info(f"REMAP (after expiry): BoltAI {current_bolt_id} -> New Grok {grok_id}")
                            grok_conv_id_for_map = grok_id; id_captured_for_map = True
                        yield new_item
                    return # Exit this wrapper after handling new conv
                
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
        else:
            response_data = await collect_full_response(final_response_gen)
            if grok_conv_id_for_map and (bolt_conv_id not in conversation_map or conversation_map[bolt_conv_id] != grok_conv_id_for_map) :
                 conversation_map[bolt_conv_id] = grok_conv_id_for_map # Ensure mapping on non-stream too
                 logger.info(f"MAP (non-stream): BoltAI {bolt_conv_id} -> Grok {grok_conv_id_for_map}")
            return JSONResponse(content={
                "id": f"chatcmpl-{response_data.get('id', 'unknown')}", "object": "chat.completion", "created": int(time.time()),
                "model": request.model or "grok-3",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": response_data.get("response","")}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": sum(len(m.get("content", "")) for m in request.messages) // 4, 
                          "completion_tokens": len(response_data.get("response","")) // 4, 
                          "total_tokens": (sum(len(m.get("content", "")) for m in request.messages) + len(response_data.get("response",""))) // 4}
            })
    except Exception as e:
        logger.error(f"Error in chat_completions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Grok (curl subprocess) API server...")
    # This script does not use .env for cookies, they are hardcoded (and need to be user-updated).
    uvicorn.run(app, host="0.0.0.0", port=8000)