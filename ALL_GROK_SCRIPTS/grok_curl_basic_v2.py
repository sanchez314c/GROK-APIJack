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
import os
from dotenv import load_dotenv

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
    cookie_parts = [
        f"x-anonuserid={anon_userid}",
        f"x-challenge={challenge}",
        f"x-signature={x_signature}",
        f"sso={sso}",
        f"cf_clearance={cf_clearance}"
    ]
    # If you have an i18nextLng or sso-rw from a very fresh cURL, you can add them here too
    # Example: cookie_parts.append(f"sso-rw={sso_rw}")
    # Example: cookie_parts.append("i18nextLng=en")

    final_cookie_string = "; ".join(cookie_parts)
    logger.debug(f"Constructed cookie string: {final_cookie_string[:100]}...") # Log beginning
    return final_cookie_string

def get_bolt_conversation_id(request: ChatCompletionRequest) -> str:
    """Extract a unique and consistent conversation ID from the BoltAI request"""
    # Check if user field contains a BoltAI conversation ID
    if request.user and request.user.startswith("bolt-"):
        logger.info(f"Using BoltAI-provided conversation ID: {request.user}")
        return request.user
        
    # Try to find the first real user message as a stable identifier
    # This should be consistent across all requests for the same conversation
    conversation_text = ""
    # Take the first 3 messages or fewer if there aren't that many
    # Skip system messages since they might change across conversations
    user_messages = [m for m in request.messages if m.get("role") == "user" and m.get("content", "")]
    if user_messages:
        # Use up to first 3 user messages to create a stable hash
        for i, msg in enumerate(user_messages[:min(3, len(user_messages))]):
            content = msg.get("content", "")
            # Skip system messages that BoltAI sometimes puts as user messages
            if not content.startswith("SYSTEM PROMPT:") and not content.startswith("Generate a concise"):
                conversation_text += content[:200]  # Just take beginning of each user message
    
    if conversation_text:
        message_hash = hashlib.md5(conversation_text.encode('utf-8')).hexdigest()
        bolt_id = f"bolt-{message_hash[:16]}"
        logger.info(f"Generated conversation ID from first user messages: {bolt_id}")
        return bolt_id
    
    # Fallback to user ID if provided (but not starting with bolt-)
    if request.user:
        return f"bolt-user-{request.user}"
    
    # Last resort - look at all message content to create a fingerprint
    message_text = ""
    for msg in request.messages:
        if "content" in msg and msg.get("role") != "system":
            message_text += msg.get("content", "")[:100]
    
    if message_text:
        message_hash = hashlib.md5(message_text.encode('utf-8')).hexdigest()
        return f"bolt-msgs-{message_hash[:16]}"
    
    # Ultimate fallback - random ID
    random_id = f"bolt-{str(uuid.uuid4())}"
    logger.info(f"Generated random conversation ID as fallback: {random_id}")
    return random_id

async def create_new_conversation(messages: List[Dict[str, Any]]) -> AsyncGenerator[Dict, None]:
    """Create a new conversation in Grok"""
    
    # Extract system prompt if it exists
    system_prompt = None
    for message in messages:
        if message.get("role") == "system" and "content" in message:
            system_prompt = message["content"]
            break
    
    # Get the latest user message
    user_message = None
    for message in reversed(messages):
        if message.get("role") == "user" and "content" in message:
            user_message = message["content"]
            break
    
    if not user_message:
        raise ValueError("No user message found")
    
    # Combine system prompt with user message for new conversations
    formatted_message = user_message
    if system_prompt:
        system_preview = system_prompt[:2000] + "..." if len(system_prompt) > 2000 else system_prompt
        formatted_message = f"SYSTEM CONTEXT: {system_preview}\n\nUSER MESSAGE: {user_message}"
    
    logger.info(f"Creating new conversation with message: {formatted_message[:100]}...")
    
    current_cookie_string = get_cookie_string()

    curl_command = [
        'curl',
        'https://grok.com/rest/app-chat/conversations/new',
        '-H', 'accept: */*',
        '-H', 'accept-language: en-US,en;q=0.5',
        '-H', 'baggage: sentry-environment=production,sentry-release=pL6qe8oXZPTmVGgCHs49g,sentry-public_key=b311e0f2690c81f25e2c4cf6d4f7ce1c,sentry-trace_id=46282aeefd494671a1de4bb48986a109,sentry-sample_rate=0,sentry-sampled=false',
        '-H', 'content-type: application/json',
        '-H', f'cookie: {current_cookie_string}',
        '-H', 'dnt: 1',
        '-H', 'origin: https://grok.com',
        '-H', 'referer: https://grok.com/',
        '-H', 'sec-ch-ua: "Brave";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
        '-H', 'sec-ch-ua-mobile: ?0',
        '-H', 'sec-ch-ua-platform: "macOS"',
        '-H', 'sec-fetch-dest: empty',
        '-H', 'sec-fetch-mode: cors',
        '-H', 'sec-fetch-site: same-origin',
        '-H', 'sec-gpc: 1',
        '-H', 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
        '--data-raw', json.dumps({
            "temporary": False,
            "modelName": "grok-3",
            "message": formatted_message,
            "fileAttachments": [],
            "imageAttachments": [],
            "disableSearch": False,
            "enableImageGeneration": True,
            "returnImageBytes": False,
            "enableImageStreaming": True,
            "imageGenerationCount": 2,
            "forceConcise": False,
            "toolOverrides": {},
            "enableSideBySide": True,
            "sendFinalMetadata": True,
            "isReasoning": False,
            "webpageUrls": [],
            "disableTextFollowUps": True
        })
    ]
    
    async for item in process_curl_response(curl_command):
        yield item

async def continue_conversation(grok_conversation_id: str, user_message: str) -> AsyncGenerator[Dict, None]:
    """Continue an existing Grok conversation"""
    
    logger.info(f"Continuing conversation {grok_conversation_id} with message: {user_message[:100]}...")
    
    current_cookie_string = get_cookie_string()

    curl_command = [
        'curl',
        f'https://grok.com/rest/app-chat/conversations/{grok_conversation_id}/messages',
        '-H', 'accept: */*',
        '-H', 'accept-language: en-US,en;q=0.5',
        '-H', 'baggage: sentry-environment=production,sentry-release=pL6qe8oXZPTmVGgCHs49g,sentry-public_key=b311e0f2690c81f25e2c4cf6d4f7ce1c,sentry-trace_id=46282aeefd494671a1de4bb48986a109,sentry-sample_rate=0,sentry-sampled=false',
        '-H', 'content-type: application/json',
        '-H', f'cookie: {current_cookie_string}',
        '-H', 'dnt: 1',
        '-H', 'origin: https://grok.com',
        '-H', 'referer: https://grok.com/',
        '-H', 'sec-ch-ua: "Brave";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
        '-H', 'sec-ch-ua-mobile: ?0',
        '-H', 'sec-ch-ua-platform: "macOS"',
        '-H', 'sec-fetch-dest: empty',
        '-H', 'sec-fetch-mode: cors',
        '-H', 'sec-fetch-site: same-origin',
        '-H', 'sec-gpc: 1',
        '-H', 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
        '--data-raw', json.dumps({
            "message": user_message,
            "fileAttachments": [],
            "imageAttachments": [],
            "disableSearch": False,
            "enableImageGeneration": True,
            "returnImageBytes": False,
            "enableImageStreaming": True,
            "imageGenerationCount": 2,
            "forceConcise": False,
            "toolOverrides": {},
            "enableSideBySide": True,
            "sendFinalMetadata": True,
            "isReasoning": False,
            "webpageUrls": [],
            "disableTextFollowUps": True
        })
    ]
    
    async for item in process_curl_response(curl_command, grok_conversation_id):
        yield item

async def process_curl_response(curl_command, conversation_id_input=None) -> AsyncGenerator[Dict, None]:
    """Process the curl response and yield tokens and messages"""
    try:
        logger.debug(f"Executing curl command: {' '.join(curl_command)[:200]}...")
        process = await asyncio.create_subprocess_exec(
            *curl_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        grok_conversation_id = conversation_id_input
        accumulated_message = ""
        response_id = str(uuid.uuid4())
        
        # Read and process stdout as it becomes available
        while True:
            line = await process.stdout.readline()
            if not line:
                break
                
            line_str = line.decode().strip()
            if not line_str:
                continue
                
            logger.debug(f"Received line from curl stdout: {line_str}")
            
            try:
                json_obj = json.loads(line_str)
                
                # Check for errors in the response
                if "code" in json_obj and json_obj.get("message") == "Not Found":
                    logger.error(f"Conversation {grok_conversation_id} not found, likely expired or invalid")
                    yield {"type": "error", "error": "Conversation not found"}
                    break
                
                # Extract conversation ID from new conversation response
                if "result" in json_obj and "conversation" in json_obj["result"]:
                    conversation_id = json_obj["result"]["conversation"].get("conversationId")
                    yield {
                        "type": "conversation_id",
                        "id": conversation_id
                    }
                
                # Extract token from streaming response
                if "result" in json_obj and "response" in json_obj["result"] and "token" in json_obj["result"]["response"]:
                    token = json_obj["result"]["response"]["token"]
                    if token:  # Skip empty tokens
                        accumulated_message += token
                        yield {
                            "type": "token",
                            "token": token,
                            "accumulated_message": accumulated_message
                        }
                
                # Extract final complete message
                if "result" in json_obj and "response" in json_obj["result"] and "modelResponse" in json_obj["result"]["response"]:
                    model_response = json_obj["result"]["response"]["modelResponse"]
                    if "message" in model_response:
                        final_message = model_response["message"]
                        yield {
                            "type": "final_message",
                            "message": final_message
                        }
                
                # For existing conversations, extract the conversation ID
                if not grok_conversation_id and "result" in json_obj and "responseId" in json_obj["result"]:
                    parent_response_id = json_obj["result"].get("parentResponseId")
                    if parent_response_id:
                        # This is from an existing conversation
                        conversation_id = parent_response_id.split('-')[0]  # Extract conversation ID from response ID
                        yield {
                            "type": "conversation_id",
                            "id": conversation_id
                        }
                        
            except json.JSONDecodeError:
                logger.warning(f"Couldn't parse line as JSON: {line_str}")
                continue
        
        # Check if process completed successfully
        await process.wait()
        if process.returncode != 0:
            stderr = await process.stderr.read()
            logger.error(f"Curl command failed: {stderr.decode()}")
            yield {"type": "error", "error": "Failed to communicate with Grok"}
            
    except Exception as e:
        logger.error(f"Unexpected error in process_curl_response: {str(e)}")
        yield {"type": "error", "error": str(e)}

async def stream_response(response_gen, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """Format the streaming response in OpenAI-compatible format"""
    response_id = f"chatcmpl-{str(uuid.uuid4())}"
    conversation_id = None
    full_message = ""
    final_message_sent = False
    error_encountered = False
    
    try:
        async for item in response_gen:
            if item["type"] == "error":
                error_encountered = True
                error_chunk = {
                    "error": {
                        "message": item["error"],
                        "type": "server_error"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                continue
                
            if item["type"] == "conversation_id":
                conversation_id = item["id"]
                continue
                
            elif item["type"] == "token":
                token = item["token"]
                full_message = item["accumulated_message"]
                
                # Create a streaming response chunk in OpenAI format
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model or "grok-3",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": token
                        },
                        "finish_reason": None
                    }]
                }
                
                # Format as Server-Sent Event
                yield f"data: {json.dumps(chunk)}\n\n"
                
            elif item["type"] == "final_message" and not final_message_sent:
                final_message_sent = True
                full_message = item["message"]
                
        # Send the final chunk to indicate completion
        if not error_encountered:
            final_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model or "grok-3",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        
        # Log the full response for debugging
        logger.info(f"Full streamed message for {response_id}: {full_message[:200]}...")
        
    except Exception as e:
        logger.error(f"Error in stream_response: {str(e)}")
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "server_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

async def collect_full_response(response_gen):
    """Collect the full response from the generator for non-streaming mode"""
    conversation_id = None
    final_message = ""
    error_message = None
    
    try:
        async for item in response_gen:
            if item["type"] == "error":
                error_message = item["error"]
                break
                
            if item["type"] == "conversation_id":
                conversation_id = item["id"]
            elif item["type"] == "final_message":
                final_message = item["message"]
                break
        
        # If we didn't get a final message, use the accumulated tokens
        if not final_message and not error_message:
            async for item in response_gen:
                if item["type"] == "token":
                    final_message = item["accumulated_message"]
        
        if error_message:
            logger.error(f"Error in collect_full_response: {error_message}")
            raise HTTPException(status_code=500, detail=error_message)
            
        logger.info(f"Extracted final message: {final_message[:200]}...")
        
        return {
            "id": conversation_id or str(uuid.uuid4()),
            "response": final_message
        }
        
    except Exception as e:
        logger.error(f"Error collecting full response: {str(e)}")
        raise

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        # Log the incoming request
        logger.debug(f"Received request with {len(request.messages)} messages")
        logger.debug(f"First few messages: {[m.get('role') for m in request.messages[:5]]}")
        
        # Get the latest user message
        user_message = None
        for message in reversed(request.messages):
            if message.get("role") == "user" and "content" in message:
                user_message = message["content"]
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found in the request")
        
        # Get BoltAI conversation ID - IMPROVED VERSION
        bolt_conversation_id = get_bolt_conversation_id(request)
        logger.info(f"BoltAI conversation ID: {bolt_conversation_id}")
        
        # Log the whole map for debugging
        logger.debug(f"Current conversation map: {conversation_map}")
        
        # Check if we've seen this conversation before
        if bolt_conversation_id in conversation_map:
            grok_conversation_id = conversation_map[bolt_conversation_id]
            logger.info(f"Continuing existing conversation {grok_conversation_id}")
            response_gen = continue_conversation(grok_conversation_id, user_message)
        else:
            logger.info("Starting new conversation")
            response_gen = create_new_conversation(request.messages)
        
        # Store conversation ID mapping if this is a new conversation
        conversation_id_captured = False
        async def wrap_response_gen():
            nonlocal conversation_id_captured
            async for item in response_gen:
                if item["type"] == "error" and "Conversation not found" in item["error"]:
                    # If the conversation is not found, start a new one
                    logger.info(f"Conversation not found, starting new conversation")
                    if bolt_conversation_id in conversation_map:
                        del conversation_map[bolt_conversation_id]  # Remove invalid mapping
                    async for new_item in create_new_conversation(request.messages):
                        if not conversation_id_captured and new_item["type"] == "conversation_id":
                            grok_id = new_item["id"]
                            conversation_map[bolt_conversation_id] = grok_id
                            logger.info(f"Mapped BoltAI conversation {bolt_conversation_id} to new Grok conversation {grok_id}")
                            conversation_id_captured = True
                        yield new_item
                    return
                if not conversation_id_captured and item["type"] == "conversation_id":
                    grok_id = item["id"]
                    conversation_map[bolt_conversation_id] = grok_id
                    logger.info(f"MAP: BoltAI {bolt_conversation_id} -> Grok {grok_id}")
                    conversation_id_captured = True
                yield item
        
        # Handle streaming mode
        if request.stream:
            return StreamingResponse(
                stream_response(wrap_response_gen(), request),
                media_type="text/event-stream"
            )
        
        # Handle non-streaming mode
        else:
            response = await collect_full_response(wrap_response_gen())
            response_text = response.get("response", "")
            
            # Estimate token counts (roughly 4 chars per token)
            prompt_tokens = sum(len(m.get("content", "")) for m in request.messages) // 4
            completion_tokens = len(response_text) // 4
            
            logger.info(f"Response from Grok: {response_text[:200]}...")
            
            return {
                "id": f"chatcmpl-{response.get('id', 'unknown')}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model or "grok-3",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
            
    except Exception as e:
        logger.error(f"Error in chat_completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Grok (curl subprocess, .env-style cookies) API server...")
    # Check if essential cookies would be loaded (or defaults used)
    cookie_string = get_cookie_string()
    if not cookie_string:
        logger.error("CRITICAL COOKIE ERROR: Cookie string is empty. Check .env or defaults.")
    uvicorn.run(app, host="0.0.0.0", port=8000)