#!/usr/bin/env python3

# Grok3 API Proxy Server
# Author: Jason Paul Michaels
# Purpose: A FastAPI-based proxy to interface with Grok's web platform using web session credentials,
#          bypassing API credit limits for unlimited usage in development environments.
# Version: Updated by Cortana with comprehensive header mimicry.

import subprocess
import pkg_resources # This line will still give a DeprecationWarning, that's okay for now
import sys
import os
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, AsyncGenerator, Generator
import httpx
import json
import asyncio
import time
import uuid
import hashlib
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import backoff

# --- Determine the script's directory for dynamic paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, '.env')
LOG_FILE_PATH = os.path.join(SCRIPT_DIR, 'grok_api.log')
CONVERSATION_MAP_PATH = os.path.join(SCRIPT_DIR, 'conversation_map.json')

# --- Function to check and install dependencies ---
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

# --- Install dependencies on script start (if run directly) ---
if __name__ == "__main__":
    install_dependencies()

# --- Load environment variables for secure credentials ---
load_dotenv(dotenv_path=ENV_PATH)

# --- Set up logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

conversation_map = {}
def save_conversation_map():
    try:
        with open(CONVERSATION_MAP_PATH, 'w') as f: json.dump({str(k): v for k, v in conversation_map.items()}, f)
        logger.info(f"Saved conversation map: {conversation_map}")
    except Exception as e: logger.error(f"Failed to save conversation map: {str(e)}")

def load_conversation_map():
    try:
        if os.path.exists(CONVERSATION_MAP_PATH):
            with open(CONVERSATION_MAP_PATH, 'r') as f: conversation_map.update(json.load(f))
            logger.info(f"Loaded conversation map: {conversation_map}")
    except Exception as e: logger.error(f"Failed to load conversation map: {str(e)}")
load_conversation_map()

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "grok-3"; messages: List[Dict[str, Any]]; temperature: Optional[float] = 1.0
    stream: Optional[bool] = False; max_tokens: Optional[int] = None; top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None; presence_penalty: Optional[float] = None
    user: Optional[str] = None; class Config: extra = "allow"

class GrokClient:
    def __init__(self):
        self.base_url = "https://grok.com/rest/app-chat/conversations"
        self.cookies = {
            "x-anonuserid": os.getenv("GROK_ANON_USERID"), "x-challenge": os.getenv("GROK_CHALLENGE"),
            "x-signature": os.getenv("GROK_X_SIGNATURE"), "sso": os.getenv("GROK_SSO"),
            "sso-rw": os.getenv("GROK_SSO_RW", os.getenv("GROK_SSO")),
            "cf_clearance": os.getenv("GROK_CF_CLEARANCE")
        }
        # --- COMPREHENSIVE HEADERS FROM CAPTURED BROWSER REQUEST ---
        self.headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.5",
            "baggage": "sentry-environment=production,sentry-release=eL61BqVW3ABxwryM6qPXq,sentry-public_key=b311e0f2690c81f25e2c4cf6d4f7ce1c,sentry-trace_id=0308f20c8f9cefc1504771b976567753,sentry-sampled=false",
            "content-type": "application/json",
            "dnt": "1",
            "origin": "https://grok.com",
            "priority": "u=1, i",
            "referer": "https://grok.com/",
            "sec-ch-ua": '"Chromium";v="136", "Brave";v="136", "Not.A/Brand";v="99"',
            "sec-ch-ua-arch": '"x86"',
            "sec-ch-ua-bitness": '"64"',
            "sec-ch-ua-full-version-list": '"Chromium";v="136.0.0.0", "Brave";v="136.0.0.0", "Not.A/Brand";v="99.0.0.0"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-model": '""',
            "sec-ch-ua-platform": '"macOS"',
            "sec-ch-ua-platform-version": '"15.0.1"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "sec-gpc": "1",
            "sentry-trace": "0308f20c8f9cefc1504771b976567753-ab4f426fc9348153-0", # Note: Sentry trace might be session/request specific
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
            "x-statsig-id": "S+K9Iv+9KokUOYyzkUD13trBrY62FTBKLoRQa7VE9HwyWjN5rCIc7e9AEb/p9kog0lEjn0hm6ReAgr9FPgUZgYa62S2MSA", # Note: Statsig ID might be session specific
            "x-xai-request-id": str(uuid.uuid4()) # Generate a new one for each request our script makes
        }
        self.timeout = httpx.Timeout(60.0, connect=10.0, read=60.0, write=10.0)

    def _prepare_payload(self, message_content: str):
        # Using the exact payload structure from the cURL command for the "Hey Grok!" message.
        return {
            "temporary":False,"modelName":"grok-3","message":message_content,
            "fileAttachments":[],"imageAttachments":[],"disableSearch":False,
            "enableImageGeneration":True,"returnImageBytes":False,
            # "returnRawGrokInXaiRequest":False, # This was in cURL, might not be needed by Grok always. Keep for now.
            "enableImageStreaming":True,"imageGenerationCount":2,"forceConcise":False,
            "toolOverrides":{},"enableSideBySide":True,"sendFinalMetadata":True,
            "isReasoning":False,"webpageUrls":[],"disableTextFollowUps":True,
            # "responseMetadata":{} # This was in cURL, likely for Grok to fill. Keep for now.
        }

    @backoff.on_exception(backoff.expo, (httpx.RequestError, httpx.HTTPStatusError), max_tries=3)
    def _stream_request(self, url: str, payload: dict) -> Generator[Dict, None, None]:
        current_headers = self.headers.copy()
        # Update dynamic headers for each request
        current_headers["x-xai-request-id"] = str(uuid.uuid4())
        # Potentially regenerate sentry-trace and baggage if they need to be fresh per request,
        # but often they are tied to a broader session or page load. For now, keep as captured.

        logger.debug(f"Streaming request to {url} with payload: {{message: {payload.get('message', '')[:50]}...}} and headers: {json.dumps({k:v for k,v in current_headers.items() if k.lower() not in ['baggage', 'cookie']}, indent=2, ensure_ascii=False)[:500]}...") # Log a subset of headers
        
        with httpx.Client(timeout=self.timeout, cookies=self.cookies, headers=current_headers) as client:
            with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                conversation_id = None; accumulated_message = ""
                for line in response.iter_lines():
                    if not line: continue
                    line_str = line.strip(); logger.debug(f"Received line: {line_str}")
                    try:
                        json_obj = json.loads(line_str)
                        if "result" in json_obj:
                            result_data = json_obj["result"]
                            if "conversation" in result_data and "conversationId" in result_data["conversation"]:
                                conversation_id = result_data["conversation"]["conversationId"]
                                logger.debug(f"EXTRACT - Found conversation ID: {conversation_id}")
                                yield {"type": "conversation_id", "id": conversation_id}
                            if "response" in result_data:
                                response_data = result_data["response"]
                                if "token" in response_data and response_data["token"]:
                                    token = response_data["token"]; accumulated_message += token
                                    yield {"type": "token", "token": token, "accumulated_message": accumulated_message}
                                if "modelResponse" in response_data and "message" in response_data["modelResponse"]:
                                    final_message = response_data["modelResponse"]["message"]
                                    logger.debug(f"EXTRACT - Found final message: {final_message[:100]}...")
                                    yield {"type": "final_message", "message": final_message}; return
                    except json.JSONDecodeError: logger.warning(f"Couldn't parse line as JSON: {line_str}")

    def new_conversation(self, message_content: str) -> Generator[Dict, None, None]:
        payload = self._prepare_payload(message_content)
        url = f"{self.base_url}/new"; yield from self._stream_request(url, payload)

    def continue_conversation(self, conversation_id: str, message_content: str) -> Generator[Dict, None, None]:
        payload = self._prepare_payload(message_content)
        url = f"{self.base_url}/{conversation_id}/messages"
        yield {"type": "conversation_id", "id": conversation_id}; yield from self._stream_request(url, payload)

grok_client = GrokClient(); executor = ThreadPoolExecutor(max_workers=10)

async def run_blocking_generator_in_thread(gen_func, *args) -> AsyncGenerator[Dict, None]:
    loop = asyncio.get_event_loop(); gen = await loop.run_in_executor(executor, lambda: list(gen_func(*args)))
    for item in gen: yield item; await asyncio.sleep(0)

async def create_new_conversation_async(messages: List[Dict[str, Any]]) -> AsyncGenerator[Dict, None]:
    system_prompts = [msg["content"] for msg in messages if msg.get("role") == "system" and msg.get("content")]
    user_messages = [msg["content"] for msg in messages if msg.get("role") == "user" and msg.get("content")]
    if not user_messages: raise ValueError("No user message found")
    user_message_content = "\n".join(user_messages)
    final_grok_message = user_message_content
    if system_prompts:
        full_sys_ctx = "\n".join(system_prompts); preview_sys_ctx = full_sys_ctx[:2000] + ("..." if len(full_sys_ctx) > 2000 else "")
        final_grok_message = f"SYSTEM CONTEXT:\n{preview_sys_ctx}\n\nUSER MESSAGE:\n{user_message_content}"
    async for item in run_blocking_generator_in_thread(grok_client.new_conversation, final_grok_message): yield item

async def continue_conversation_async(grok_conv_id: str, user_msg_content: str) -> AsyncGenerator[Dict, None]:
    async for item in run_blocking_generator_in_thread(grok_client.continue_conversation, grok_conv_id, user_msg_content): yield item

async def format_openai_stream_response(response_gen: AsyncGenerator[Dict, None], req_model: Optional[str]) -> AsyncGenerator[str, None]:
    resp_id = f"chatcmpl-{uuid.uuid4()}"; created_time = int(time.time())
    async for item in response_gen:
        if item["type"] == "token":
            chunk = {"id": resp_id, "object": "chat.completion.chunk", "created": created_time,
                       "model": req_model or "grok-3", "choices": [{"index": 0, "delta": {"content": item["token"]}, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n"
        elif item["type"] == "final_message": break
    final_chunk = {"id": resp_id, "object": "chat.completion.chunk", "created": created_time,
                   "model": req_model or "grok-3", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    yield f"data: {json.dumps(final_chunk)}\n\n"; yield "data: [DONE]\n\n"

async def collect_full_openai_response(response_gen: AsyncGenerator[Dict, None], req_model: Optional[str], req_msgs: List[Dict[str,Any]]):
    full_resp_content = ""; grok_conv_id_for_resp = None
    async for item in response_gen:
        if item["type"] == "conversation_id": grok_conv_id_for_resp = item["id"]
        elif item["type"] == "token": full_resp_content = item["accumulated_message"]
        elif item["type"] == "final_message": full_resp_content = item["message"]; break
    prompt_toks = sum(len(m.get("content", "")) for m in req_msgs) // 4
    completion_toks = len(full_resp_content) // 4
    return {"id": f"chatcmpl-{grok_conv_id_for_resp or uuid.uuid4()}", "object": "chat.completion", "created": int(time.time()),
            "model": req_model or "grok-3", "choices": [{"index": 0, "message": {"role": "assistant", "content": full_resp_content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_toks, "completion_tokens": completion_toks, "total_tokens": prompt_toks + completion_toks}}

def get_bolt_conversation_id(request: ChatCompletionRequest) -> str:
    user_msgs_content = "".join(msg.get("content","")[:200] for msg in request.messages if msg.get("role")=="user" and "content" in msg and not msg["content"].lower().startswith(("system prompt:","generate a concise","user:","system:")))
    if user_msgs_content: return f"bolt-{hashlib.md5(user_msgs_content.encode('utf-8')).hexdigest()[:16]}"
    if request.user: return f"bolt-user-{request.user}"
    return f"bolt-fallback-{uuid.uuid4()}"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        user_msgs_for_grok = [msg for msg in request.messages if msg.get("role") == "user"]
        if not user_msgs_for_grok or not user_msgs_for_grok[-1].get("content"):
            raise HTTPException(status_code=400, detail="No valid user message content found")
        last_user_msg_content = user_msgs_for_grok[-1]["content"]
        bolt_conv_id = get_bolt_conversation_id(request); logger.info(f"BoltAI conversation ID: {bolt_conv_id}")
        grok_resp_gen: Optional[AsyncGenerator[Dict, None]] = None; grok_conv_id_to_map = None

        if bolt_conv_id in conversation_map:
            grok_conv_id_from_map = conversation_map[bolt_conv_id]
            logger.info(f"CONTINUATION: Using Grok conv {grok_conv_id_from_map} for Bolt ID {bolt_conv_id}")
            grok_resp_gen = continue_conversation_async(grok_conv_id_from_map, last_user_msg_content)
            grok_conv_id_to_map = grok_conv_id_from_map
        else:
            logger.info(f"NEW CONVERSATION: Starting Grok conv for Bolt ID {bolt_conv_id}")
            grok_resp_gen = create_new_conversation_async(request.messages)
        if grok_resp_gen is None: raise HTTPException(status_code=500, detail="Failed to init Grok conv generator")

        async def stream_wrapper_for_mapping(gen: AsyncGenerator[Dict, None], curr_bolt_id: str) -> AsyncGenerator[Dict, None]:
            nonlocal grok_conv_id_to_map; id_captured = False
            async for item in gen:
                if not id_captured and item["type"] == "conversation_id":
                    new_grok_id = item["id"]
                    if curr_bolt_id not in conversation_map or conversation_map[curr_bolt_id] != new_grok_id:
                        conversation_map[curr_bolt_id] = new_grok_id
                        logger.info(f"MAP: BoltAI {curr_bolt_id} -> Grok {new_grok_id}"); save_conversation_map()
                    grok_conv_id_to_map = new_grok_id; id_captured = True
                yield item
        
        processed_gen = stream_wrapper_for_mapping(grok_resp_gen, bolt_conv_id)
        if request.stream:
            return StreamingResponse(format_openai_stream_response(processed_gen, request.model), media_type="text/event-stream")
        else:
            full_resp_data = await collect_full_openai_response(processed_gen, request.model, request.messages)
            if grok_conv_id_to_map and (bolt_conv_id not in conversation_map or conversation_map[bolt_conv_id] != grok_conv_id_to_map):
                 conversation_map[bolt_conv_id] = grok_conv_id_to_map
                 logger.info(f"MAP (non-stream): BoltAI {bolt_conv_id} -> Grok {grok_conv_id_to_map}"); save_conversation_map()
            return JSONResponse(content=full_resp_data)
    except Exception as e:
        logger.error(f"Error in chat_completions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Grok API server from: {SCRIPT_DIR}, .env at: {ENV_PATH}")
    if not all(os.getenv(k) for k in ["GROK_CF_CLEARANCE", "GROK_SSO", "GROK_CHALLENGE", "GROK_X_SIGNATURE"]):
        logger.warning(f".env file at {ENV_PATH} missing one or more key Grok cookies. API will likely fail.")
        print(f"WARNING: .env file at {ENV_PATH} is missing key Grok cookies. Ensure it's created with fresh values.")
    try: uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e: logger.error(f"Failed to start Uvicorn: {str(e)}"); print(f"ERROR: {str(e)}"); sys.exit(1)