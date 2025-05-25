#!/usr/bin/env python3

# Grok3 API Proxy Server
# Author: Jason Paul Michaels
# Purpose: A FastAPI-based proxy to interface with Grok's web platform using web session credentials,
#          bypassing API credit limits for unlimited usage in development environments.
# Version: Updated by Cortana for timeout fixes, auto-setup, and auto-run on 2025-05-12

import subprocess
import pkg_resources
import sys
import os
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Generator, AsyncGenerator
import httpx
import json
import asyncio
import time
import uuid
import hashlib
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import backoff

# Function to check and install dependencies
def install_dependencies():
    """Check for required packages and install them if missing."""
    required = ['fastapi', 'uvicorn', 'httpx', 'pydantic', 'python-dotenv', 'backoff']
    missing = []
    for pkg in required:
        try:
            pkg_resources.get_distribution(pkg)
        except pkg_resources.DistributionNotFound:
            missing.append(pkg)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}. Installing now...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            print("All dependencies installed successfully.")
        except subprocess.CalledProcessError:
            print("Failed to install dependencies. Please install manually with:")
            print(f"pip install {' '.join(missing)}")
            sys.exit(1)
    else:
        print("All required dependencies are already installed.")

# Install dependencies on script start
if __name__ == "__main__":
    install_dependencies()

# Load environment variables for secure credentials
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a file handler for persistent logs
file_handler = logging.FileHandler('grok_api.log')
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory conversation store with persistence to disk
conversation_map = {}

def save_conversation_map():
    """Save conversation map to disk for persistence across restarts."""
    try:
        with open('conversation_map.json', 'w') as f:
            serializable_map = {str(k): v for k, v in conversation_map.items()}
            json.dump(serializable_map, f)
        logger.info(f"Saved conversation map to disk: {conversation_map}")
    except Exception as e:
        logger.error(f"Failed to save conversation map: {str(e)}")

def load_conversation_map():
    """Load conversation map from disk if it exists."""
    try:
        if os.path.exists('conversation_map.json'):
            with open('conversation_map.json', 'r') as f:
                loaded_map = json.load(f)
                conversation_map.update(loaded_map)
            logger.info(f"Loaded conversation map from disk: {conversation_map}")
    except Exception as e:
        logger.error(f"Failed to load conversation map: {str(e)}")

# Load the map on startup
load_conversation_map()

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

class GrokClient:
    def __init__(self):
        self.base_url = "https://grok.com/rest/app-chat/conversations"
        # Load credentials from environment variables or fallback to hardcoded values
        # IMPORTANT: Update these with fresh cookies if they expire. See instructions in script comments or README.
        self.cookies = {
            "x-anonuserid": os.getenv("GROK_ANON_USERID", "e63622de-bd3e-4f65-9cd5-e71a4bb42100"),
            "x-challenge": os.getenv("GROK_CHALLENGE", "ULttT8xKoBKE0ngRbDCxxDwT2tiW8mtTR2iR9Kde8HqyrODEcXw0X0nUgUeeLYU56sQbLPjXbrDFgB8GaIio+XjjndC1B6p9tgvdnLiC9vRhDm7CocyKliu2BBcwvOo/j/pxie1dE3hr2zx4BneBq3LUCb7rg1WRLpf3sd71bjKt9EUi7p4="),
            "x-signature": os.getenv("GROK_SIGNATURE", "764WcSQwtWquxSHWKA+hekn4e3VA64ET882DakagE4V9RZSG0Z/dfCMe0LXK1ZlOkmk0ZehdD/2L0Hd+1QZtqQ=="),
            "sso": os.getenv("GROK_SSO", "eyJhbGciOiJIUzI1NiJ9.eyJzZXNzaW9uX2lkIjoiNTFjYTdiOGUtZDQ0YS00MTA0LWI1NWQtMjdlNjkwNTlkMGM3In0.JMikOs2HQQipXxTcCAf9ETb6-exnzcdcoXYyBPkG9ek"),
            "cf_clearance": os.getenv("GROK_CF_CLEARANCE", "VdJ8R6oTcB97z5rjEV8KoPv0C.y3ZAbpDK5W_sgqGIU-1744162953-1.2.1.1-_ZYib7Qy8MLN6dV0UrcBeZZ6QmhPnlsVsner_A6SM5hPCXTGLxJ_mc6QgkaOm6W3z3VA9NR6MKDwH6SnGO_q1cgDeJN5B37E8ViqDHbkPI5YOsfLX0Wc82JFGLlWfmvb94B5_AZKVmkhjIoDQTwISQpCzY8K.C1KLOmvArtrJIahijuZG8lcMy0yHHCvncyEsL0ySqE6V_l3mjgdq.ZfSsm4gsBel6sum4pa_R3czRjjN3CgJM3Mcw31SAXRr.Tqwvh1neHUzWh_oD.ViPIdh7PthCBbaE8BdjAlBeWP4k.GxjeWHwv5mDNTPSvgGxCtqsFsVr0b.X2IS4vt2HfZjsaBIt309Yfb.F1hCBwGxM8")
        }
        self.headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.5",
            "content-type": "application/json",
            "origin": "https://grok.com",
            "referer": "https://grok.com/",
            "sec-ch-ua": '"Brave";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "sec-gpc": "1",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
        }
        # Timeout configuration for HTTP requests
        self.timeout = httpx.Timeout(60.0, connect=10.0, read=60.0, write=10.0)

    def _prepare_payload(self, message):
        """Prepare the default payload with the user's message."""
        return {
            "temporary": False,
            "modelName": "grok-3",
            "message": message,
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
        }

    @backoff.on_exception(backoff.expo, (httpx.RequestError, httpx.HTTPStatusError), max_tries=3)
    def new_conversation(self, message):
        """Start a new conversation with Grok with retry mechanism."""
        logger.info(f"Creating new conversation with message: {message[:100]}...")
        
        payload = self._prepare_payload(message)
        url = f"{self.base_url}/new"
        
        with httpx.Client(timeout=self.timeout, cookies=self.cookies, headers=self.headers) as client:
            with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                conversation_id = None
                accumulated_message = ""
                
                for line in response.iter_lines():
                    if not line:
                        continue
                        
                    line_str = line.strip()
                    logger.debug(f"Received line: {line_str}")
                    
                    try:
                        json_obj = json.loads(line_str)
                        
                        if "result" in json_obj and "conversation" in json_obj["result"]:
                            conversation_id = json_obj["result"]["conversation"].get("conversationId")
                            logger.info(f"EXTRACT - Found conversation ID: {conversation_id}")
                            yield {
                                "type": "conversation_id",
                                "id": conversation_id
                            }
                        
                        if "result" in json_obj and "response" in json_obj["result"] and "token" in json_obj["result"]["response"]:
                            token = json_obj["result"]["response"]["token"]
                            if token:
                                accumulated_message += token
                                yield {
                                    "type": "token",
                                    "token": token,
                                    "accumulated_message": accumulated_message
                                }
                        
                        if "result" in json_obj and "response" in json_obj["result"] and "modelResponse" in json_obj["result"]["response"]:
                            model_response = json_obj["result"]["response"]["modelResponse"]
                            if "message" in model_response:
                                final_message = model_response["message"]
                                logger.info(f"EXTRACT - Found final message: {final_message[:100]}...")
                                yield {
                                    "type": "final_message",
                                    "message": final_message
                                }
                    except json.JSONDecodeError:
                        logger.warning(f"Couldn't parse line as JSON: {line_str}")
                        continue

    @backoff.on_exception(backoff.expo, (httpx.RequestError, httpx.HTTPStatusError), max_tries=3)
    def continue_conversation(self, conversation_id, message):
        """Continue an existing conversation with Grok with retry mechanism."""
        logger.info(f"CONTINUATION: Continuing conversation {conversation_id} with message: {message[:100]}...")
        
        payload = self._prepare_payload(message)
        url = f"{self.base_url}/{conversation_id}/messages"
        
        with httpx.Client(timeout=self.timeout, cookies=self.cookies, headers=self.headers) as client:
            with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                accumulated_message = ""
                
                for line in response.iter_lines():
                    if not line:
                        continue
                        
                    line_str = line.strip()
                    logger.debug(f"Received line: {line_str}")
                    
                    try:
                        json_obj = json.loads(line_str)
                        
                        if "result" in json_obj and "response" in json_obj["result"] and "token" in json_obj["result"]["response"]:
                            token = json_obj["result"]["response"]["token"]
                            if token:
                                accumulated_message += token
                                yield {
                                    "type": "token",
                                    "token": token,
                                    "accumulated_message": accumulated_message
                                }
                        
                        if "result" in json_obj and "response" in json_obj["result"] and "modelResponse" in json_obj["result"]["response"]:
                            model_response = json_obj["result"]["response"]["modelResponse"]
                            if "message" in model_response:
                                final_message = model_response["message"]
                                logger.info(f"EXTRACT - Found final message: {final_message[:100]}...")
                                yield {
                                    "type": "final_message",
                                    "message": final_message
                                }
                        
                        if not accumulated_message and not json_obj.get("result", {}).get("response", {}).get("modelResponse"):
                            yield {
                                "type": "conversation_id",
                                "id": conversation_id
                            }
                    except json.JSONDecodeError:
                        logger.warning(f"Couldn't parse line as JSON: {line_str}")
                        continue

# Create a shared client instance
grok_client = GrokClient()

# Executor for running blocking requests in a thread pool
executor = ThreadPoolExecutor(max_workers=10)

async def process_in_thread(func, *args):
    """Run a blocking function in a thread pool and yield results through asyncio."""
    loop = asyncio.get_event_loop()
    gen = await loop.run_in_executor(executor, func, *args)
    for item in gen:
        yield item

async def create_new_conversation(messages: List[Dict[str, Any]]) -> AsyncGenerator[Dict, None]:
    """Create a new conversation in Grok."""
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
    
    async for item in process_in_thread(grok_client.new_conversation, formatted_message):
        yield item

async def continue_conversation(grok_conversation_id: str, user_message: str) -> AsyncGenerator[Dict, None]:
    """Continue an existing Grok conversation."""
    async for item in process_in_thread(grok_client.continue_conversation, grok_conversation_id, user_message):
        yield item

async def stream_response(response_gen, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """Format the streaming response in OpenAI-compatible format."""
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
                yield f"data: {json.dumps(chunk)}\n\n"
                
            elif item["type"] == "final_message" and not final_message_sent:
                final_message_sent = True
                full_message = item["message"]
                
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
            logger.info(f"Full streamed message: {full_message}")
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
    """Collect the full response from the generator for non-streaming mode."""
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
        
        if not final_message and not error_message:
            async for item in response_gen:
                if item["type"] == "token":
                    final_message = item["accumulated_message"]
        
        if error_message:
            logger.error(f"Error in collect_full_response: {error_message}")
            raise HTTPException(status_code=500, detail=error_message)
            
        logger.info(f"Extracted final message: {final_message}")
        return {
            "id": conversation_id or str(uuid.uuid4()),
            "response": final_message
        }
    except Exception as e:
        logger.error(f"Error collecting full response: {str(e)}")
        raise

def get_bolt_conversation_id(request: ChatCompletionRequest) -> str:
    """Extract a unique conversation ID from the BoltAI request."""
    logger.debug(f"Message roles: {[m.get('role') for m in request.messages]}")
    first_user_message = None
    for msg in request.messages:
        if msg.get("role") == "user" and "content" in msg:
            content = msg.get("content", "")
            if not content.startswith("SYSTEM PROMPT:") and not content.startswith("Generate a concise"):
                first_user_message = content
                break
    
    if first_user_message:
        message_hash = hashlib.md5(first_user_message.encode('utf-8')).hexdigest()
        conversation_id = f"bolt-{message_hash[:16]}"
        logger.info(f"Generated conversation ID from first user message: {conversation_id}")
        return conversation_id
    
    if request.user:
        return f"bolt-user-{request.user}"
    
    message_text = ""
    for msg in request.messages:
        if "content" in msg and msg.get("role") != "system":
            message_text += msg.get("content", "")[:100]
    
    if message_text:
        message_hash = hashlib.md5(message_text.encode('utf-8')).hexdigest()
        return f"bolt-msgs-{message_hash[:16]}"
    
    return f"bolt-{str(uuid.uuid4())}"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle incoming chat completion requests and route to Grok."""
    try:
        logger.debug(f"Received request with {len(request.messages)} messages")
        logger.debug(f"First few messages: {[m.get('role') for m in request.messages[:5]]}")
        
        user_message = None
        for message in reversed(request.messages):
            if message.get("role") == "user" and "content" in message:
                user_message = message["content"]
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found in the request")
        
        bolt_conversation_id = get_bolt_conversation_id(request)
        logger.info(f"BoltAI conversation ID: {bolt_conversation_id}")
        logger.debug(f"Current conversation map: {conversation_map}")
        
        if bolt_conversation_id in conversation_map:
            grok_conversation_id = conversation_map[bolt_conversation_id]
            logger.info(f"CONTINUATION: Using existing conversation {grok_conversation_id}")
            response_gen = continue_conversation(grok_conversation_id, user_message)
        else:
            logger.info("NEW CONVERSATION: Starting new conversation")
            response_gen = create_new_conversation(request.messages)
        
        conversation_id_captured = False
        async def wrap_response_gen():
            nonlocal conversation_id_captured
            async for item in response_gen:
                if not conversation_id_captured and item["type"] == "conversation_id":
                    grok_id = item["id"]
                    conversation_map[bolt_conversation_id] = grok_id
                    logger.info(f"MAP: BoltAI conversation {bolt_conversation_id} -> Grok conversation {grok_id}")
                    conversation_id_captured = True
                    save_conversation_map()
                yield item
        
        if request.stream:
            return StreamingResponse(
                stream_response(wrap_response_gen(), request),
                media_type="text/event-stream"
            )
        else:
            response = await collect_full_response(wrap_response_gen())
            response_text = response.get("response", "")
            prompt_tokens = sum(len(m.get("content", "")) for m in request.messages) // 4
            completion_tokens = len(response_text) // 4
            
            logger.info(f"Response from Grok: {response_text}")
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
    logger.info("Starting Grok API server...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"Failed to start server on port 8000: {str(e)}")
        print(f"Failed to start server on port 8000: {str(e)}")
        print("Please ensure port 8000 is free or modify the port in the script.")
        sys.exit(1)