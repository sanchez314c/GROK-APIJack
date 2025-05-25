#!/usr/bin/env python3

import subprocess
import sys
import os
import logging
from logging.handlers import RotatingFileHandler
import asyncio
import time
import uuid
import hashlib
import json
from typing import List, Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

# --- Auto-detect and install dependencies ---
def install_package(package_name):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "show", package_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"{package_name} is already installed.")
    except subprocess.CalledProcessError:
        logger.info(f"Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info(f"Successfully installed {package_name}.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package_name}: {e}")
            sys.exit(1)

def install_playwright_browsers():
    try:
        subprocess.check_call([sys.executable, "-m", "playwright", "install"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Playwright browsers installed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install Playwright browsers: {e}")
        sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proxy.log")
handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Install required packages
required_packages = ["fastapi", "pydantic", "uvicorn", "python-dotenv", "playwright"]
for package in required_packages:
    install_package(package)
install_playwright_browsers()

# Import dependencies
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv, set_key
from playwright.async_api import async_playwright

# --- Paths and environment setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, '.env')
load_dotenv(dotenv_path=ENV_PATH)

GOOGLE_EMAIL = os.getenv('GOOGLE_EMAIL')
GOOGLE_PASSWORD = os.getenv('GOOGLE_PASSWORD')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File paths for conversation and session maps
CONVERSATION_MAP_FILE = os.path.join(SCRIPT_DIR, "conversation_map.json")
SESSION_MAP_FILE = os.path.join(SCRIPT_DIR, "session_map.json")

# Global maps
conversation_map = {}
def load_conversation_map():
    global conversation_map
    if os.path.exists(CONVERSATION_MAP_FILE):
        try:
            with open(CONVERSATION_MAP_FILE, 'r') as f:
                loaded_map = json.load(f)
                for bolt_id, data in loaded_map.items():
                    if isinstance(data, str):
                        conversation_map[bolt_id] = {"conversation_id": data, "last_response_id": None, "message_sequence": 0}
                    else:
                        data["message_sequence"] = data.get("message_sequence", 0)
                        conversation_map[bolt_id] = data
                logger.info(f"Loaded {len(conversation_map)} conversation mappings from file")
        except Exception as e:
            logger.error(f"Error loading conversation map: {e}")
load_conversation_map()

session_map = {}
def load_session_map():
    global session_map
    if os.path.exists(SESSION_MAP_FILE):
        try:
            with open(SESSION_MAP_FILE, 'r') as f:
                session_map = json.load(f)
                logger.info(f"Loaded {len(session_map)} session mappings from file")
        except Exception as e:
            logger.error(f"Error loading session map: {e}")
load_session_map()

def save_conversation_map():
    try:
        with open(CONVERSATION_MAP_FILE, 'w') as f:
            json.dump(conversation_map, f)
            logger.debug(f"Saved {len(conversation_map)} conversation mappings to file")
    except Exception as e:
        logger.error(f"Error saving conversation map: {e}")

def save_session_map():
    try:
        with open(SESSION_MAP_FILE, 'w') as f:
            json.dump(session_map, f)
            logger.debug(f"Saved {len(session_map)} session mappings to file")
    except Exception as e:
        logger.error(f"Error saving session map: {e}")

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
    cf_clearance = os.getenv('GROK_CF_CLEARANCE')
    sso = os.getenv('GROK_SSO')
    anon_userid = os.getenv('GROK_ANON_USERID')
    challenge = os.getenv('GROK_CHALLENGE')
    x_signature = os.getenv('GROK_X_SIGNATURE')

    if not all([cf_clearance, sso, anon_userid, challenge, x_signature]):
        logger.error("CRITICAL: One or more essential cookie values are missing.")
        return ""

    cookie_parts = [
        f"x-anonuserid={anon_userid}",
        f"x-challenge={challenge}",
        f"x-signature={x_signature}",
        f"sso={sso}",
        f"cf_clearance={cf_clearance}"
    ]

    final_cookie_string = "; ".join(cookie_parts)
    logger.debug(f"Constructed cookie string: {final_cookie_string[:100]}...")
    return final_cookie_string

async def check_cookies_validity() -> bool:
    cookie_string = get_cookie_string()
    if not cookie_string:
        logger.warning("Cookie string is empty.")
        return False

    curl_command = [
        'curl',
        'https://grok.com/rest/app-chat/conversations',
        '-H', 'accept: */*',
        '-H', 'accept-language: en-US,en;q=0.5',
        '-H', 'content-type: application/json',
        '-H', f'cookie: {cookie_string}',
        '-H', 'origin: https://grok.com',
        '-H', 'referer: https://grok.com/',
        '-H', 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *curl_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()
        response = stdout.decode().strip()
        logger.debug(f"Cookie validity check response: {response[:100]}...")

        if process.returncode != 0:
            logger.warning(f"Cookie validity check failed: {stderr.decode()[:200]}...")
            return False

        try:
            data = json.loads(response)
            if "code" in data and data["code"] in [401, 403, 5]:
                logger.warning("Cookies are invalid (unauthorized or session expired).")
                return False
            logger.info("Cookies are valid.")
            return True
        except json.JSONDecodeError:
            logger.error(f"Failed to parse cookie validity response: {response[:100]}...")
            return False
    except Exception as e:
        logger.error(f"Error checking cookie validity: {str(e)}")
        return False

async def refresh_cookies() -> bool:
    logger.info("Attempting to refresh cookies via Playwright using Google SSO...")
    
    if not GOOGLE_EMAIL or not GOOGLE_PASSWORD:
        logger.error("Google credentials not found in .env. Cannot refresh cookies.")
        return False

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()

            logger.info("Navigating to GrokWWW...")
            await page.goto('https://grok.com')

            logger.info('Clicking "Sign in with X"...')
            await page.click('text=Sign in with X')
            await page.wait_for_navigation()

            logger.info("Attempting to log in via Google SSO...")
            await page.click('button[data-testid="OAuth_Provider_Google"]')
            await page.wait_for_navigation()

            if "accounts.google.com" in page.url and "ServiceLogin" in page.url:
                logger.info("Entering Google email...")
                await page.fill('input[type="email"]', GOOGLE_EMAIL)
                await page.click('button:has-text("Next")')
                await asyncio.sleep(1)

                logger.info("Entering Google password...")
                await page.fill('input[type="password"]', GOOGLE_PASSWORD)
                await page.click('button:has-text("Next")')
                await asyncio.sleep(1)
            elif "accounts.google.com" in page.url and "identifier" not in page.url:
                logger.info("Google account selection page detected. Selecting account...")
                await page.click(f'div[data-identifier="{GOOGLE_EMAIL}"]')
                await asyncio.sleep(1)

            try:
                await page.wait_for_url('https://x.com/**', timeout=10000)
                logger.info("Successfully authenticated with Google SSO, redirected to X...")
            except Exception as e:
                logger.error(f"Google SSO flow failed, possibly due to 2FA or other prompts: {e}")
                await browser.close()
                return False

            logger.info("Waiting for redirect to GrokWWW...")
            await page.wait_for_url('https://grok.com/**', timeout=30000)

            logger.info("Extracting cookies...")
            cookies = await context.cookies('https://grok.com')
            cookie_map = {cookie['name']: cookie['value'] for cookie in cookies}

            required_cookies = {
                'GROK_CF_CLEARANCE': cookie_map.get('cf_clearance', ''),
                'GROK_SSO': cookie_map.get('sso', ''),
                'GROK_ANON_USERID': cookie_map.get('x-anonuserid', ''),
                'GROK_CHALLENGE': cookie_map.get('x-challenge', ''),
                'GROK_X_SIGNATURE': cookie_map.get('x-signature', '')
            }

            logger.info("Updating .env file with new cookies...")
            for key, value in required_cookies.items():
                set_key(ENV_PATH, key, value)
            
            load_dotenv(dotenv_path=ENV_PATH, override=True)
            logger.info("Cookies refreshed and .env updated successfully!")
            await browser.close()
            return True
    except Exception as e:
        logger.error(f"Failed to refresh cookies: {e}")
        return False

async def validate_conversation(conversation_id: str) -> bool:
    logger.info(f"Validating conversation ID: {conversation_id}")
    cookie_string = get_cookie_string()
    if not cookie_string:
        logger.warning("Cookie string is empty. Attempting to refresh cookies...")
        success = await refresh_cookies()
        if not success:
            logger.error("Failed to refresh cookies. Cannot validate conversation.")
            return False
        cookie_string = get_cookie_string()
        if not cookie_string:
            logger.error("Still no valid cookies after refresh.")
            return False

    if not await check_cookies_validity():
        logger.warning("Cookies are invalid. Attempting to refresh...")
        success = await refresh_cookies()
        if not success:
            logger.error("Failed to refresh cookies. Cannot validate conversation.")
            return False
        cookie_string = get_cookie_string()
        if not cookie_string:
            logger.error("Still no valid cookies after refresh.")
            return False

    curl_command = [
        'curl',
        f'https://grok.com/rest/app-chat/conversations/{conversation_id}',
        '-H', 'accept: */*',
        '-H', 'accept-language: en-US,en;q=0.5',
        '-H', 'content-type: application/json',
        '-H', f'cookie: {cookie_string}',
        '-H', 'origin: https://grok.com',
        '-H', 'referer: https://grok.com/',
        '-H', 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
    ]

    max_attempts = 3
    for attempt in range(max_attempts):
        logger.info(f"Validation attempt {attempt + 1}/{max_attempts} for conversation {conversation_id}")
        process = await asyncio.create_subprocess_exec(
            *curl_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()
        response = stdout.decode().strip()
        logger.debug(f"Validation response (attempt {attempt + 1}): {response[:100]}...")

        try:
            data = json.loads(response)
            if "code" in data and data["code"] == 5:
                logger.warning(f"Conversation {conversation_id} not found (code 5) on attempt {attempt + 1}. Assuming reusable unless auth fails.")
                return True  # Assume reusable to prevent new convo flood
            elif "code" in data and data["code"] in [401, 403]:
                logger.warning(f"Auth error (code {data['code']}) on attempt {attempt + 1}, refreshing cookies...")
                if attempt < max_attempts - 1:
                    success = await refresh_cookies()
                    if success:
                        cookie_string = get_cookie_string()
                        curl_command[4] = f'cookie: {cookie_string}'
                        await asyncio.sleep(2)
                        continue
                logger.error(f"Persistent auth failure for {conversation_id} after {max_attempts} attempts")
                return False
            logger.info(f"Conversation {conversation_id} validated successfully on attempt {attempt + 1}")
            return True
        except json.JSONDecodeError:
            logger.error(f"Failed to parse validation response (attempt {attempt + 1}): {response[:100]}...")
            if attempt < max_attempts - 1:
                await asyncio.sleep(2)
                continue
            logger.warning(f"Parse error persisted, assuming conversation {conversation_id} is reusable to avoid flooding.")
            return True  # Default to reusable to prevent flooding
    logger.error(f"Validation inconclusive for {conversation_id} after {max_attempts} attempts, defaulting to reusable.")
    return True  # Err on the side of reuse

def get_bolt_conversation_id(request: ChatCompletionRequest, client_ip: str) -> str:
    logger.debug(f"Request user field: {request.user}")
    logger.debug(f"Request messages: {[msg.get('role') + ': ' + str(msg.get('content', ''))[:50] for msg in request.messages]}")
    logger.debug(f"Client IP: {client_ip}")
    
    if request.user and request.user.startswith("bolt-"):
        logger.info(f"Using BoltAI-provided conversation ID: {request.user}")
        return request.user
    
    if client_ip in session_map:
        bolt_id = session_map[client_ip]
        logger.info(f"Reusing session ID from session_map for IP {client_ip}: {bolt_id}")
        return bolt_id
    
    bolt_id = f"bolt-ip-{client_ip}-persistent"
    session_map[client_ip] = bolt_id
    save_session_map()
    logger.info(f"Generated stable session ID for IP {client_ip} and saved to session_map: {bolt_id}")
    return bolt_id

async def create_new_conversation(messages: List[Dict[str, Any]], is_retry: bool = False) -> AsyncGenerator[Dict, None]:
    filtered_messages = []
    for msg in messages:
        if msg.get("role") == "system" and "json" in msg.get("content", "").lower():
            logger.info("Filtering out system prompt containing JSON instruction")
            continue
        filtered_messages.append(msg)

    system_prompt = None
    user_messages = []
    for message in filtered_messages:
        if message.get("role") == "system" and "content" in message:
            system_prompt = message["content"]
        elif message.get("role") == "user" and "content" in message:
            user_messages.append(message["content"])
    
    if not user_messages:
        raise ValueError("No user message found")
    
    formatted_message = user_messages[-1]
    if system_prompt and not is_retry:
        formatted_message = f"SYSTEM: {system_prompt}\n\nUSER: {formatted_message}"
    logger.info(f"Creating new conversation with message: {formatted_message[:100]}...")
    
    current_cookie_string = get_cookie_string()
    if not current_cookie_string:
        logger.warning("Cookie string is empty. Attempting to refresh cookies...")
        success = await refresh_cookies()
        if not success:
            logger.error("Failed to refresh cookies. Cannot create new conversation.")
            yield {"type": "error", "error": "Failed to refresh cookies"}
            return
        current_cookie_string = get_cookie_string()

    if not await check_cookies_validity():
        logger.warning("Cookies are invalid. Attempting to refresh...")
        success = await refresh_cookies()
        if not success:
            logger.error("Failed to refresh cookies. Cannot create new conversation.")
            yield {"type": "error", "error": "Failed to refresh cookies"}
            return
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

async def continue_conversation(grok_conversation_id: str, messages: List[Dict[str, Any]], parent_response_id: Optional[str] = None) -> AsyncGenerator[Dict, None]:
    filtered_messages = []
    for msg in messages:
        if msg.get("role") == "system" and "json" in msg.get("content", "").lower():
            logger.info("Filtering out system prompt containing JSON instruction")
            continue
        filtered_messages.append(msg)

    user_messages = []
    for message in filtered_messages:
        if message.get("role") == "user" and "content" in message:
            user_messages.append(message["content"])
    
    if not user_messages:
        raise ValueError("No user message found")
    
    formatted_message = user_messages[-1]
    logger.info(f"Continuing conversation {grok_conversation_id} with message: {formatted_message[:100]}... Parent Response ID: {parent_response_id}")
    
    current_cookie_string = get_cookie_string()
    if not current_cookie_string:
        logger.warning("Cookie string is empty. Attempting to refresh cookies...")
        success = await refresh_cookies()
        if not success:
            logger.error("Failed to refresh cookies. Cannot continue conversation.")
            yield {"type": "error", "error": "Failed to refresh cookies"}
            return
        current_cookie_string = get_cookie_string()

    if not await check_cookies_validity():
        logger.warning("Cookies are invalid. Attempting to refresh...")
        success = await refresh_cookies()
        if not success:
            logger.error("Failed to refresh cookies. Cannot continue conversation.")
            yield {"type": "error", "error": "Failed to refresh cookies"}
            return
        current_cookie_string = get_cookie_string()

    payload = {
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
    }
    if parent_response_id:
        payload["parentResponseId"] = parent_response_id
    else:
        logger.warning("No parentResponseId available for continuing conversation")

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
        '--data-raw', json.dumps(payload)
    ]
    
    async for item in process_curl_response(curl_command, grok_conversation_id):
        yield item

async def process_curl_response(curl_command, conversation_id_input=None) -> AsyncGenerator[Dict, None]:
    for attempt in range(3):
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
            
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                    
                line_str = line.decode().strip()
                if not line_str:
                    continue
                    
                logger.debug(f"Received line from curl stdout: {line_str[:200]}...")
                
                try:
                    json_obj = json.loads(line_str)
                    
                    if "code" in json_obj and json_obj.get("message") == "Not Found":
                        error_msg = f"Conversation {grok_conversation_id} not found, likely expired or invalid"
                        logger.error(error_msg)
                        yield {"type": "error", "error": error_msg}
                        break
                    
                    if "result" in json_obj and "conversation" in json_obj["result"]:
                        conversation_id = json_obj["result"]["conversation"].get("conversationId")
                        if conversation_id:
                            logger.info(f"Extracted new conversation ID: {conversation_id}")
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
                            response_id = model_response.get("responseId")
                            if response_id:
                                logger.info(f"Extracted response ID: {response_id}")
                            logger.info(f"Received final message: {final_message[:100]}...")
                            yield {
                                "type": "final_message",
                                "message": final_message,
                                "response_id": response_id if response_id else str(uuid.uuid4())
                            }
                    
                    if not grok_conversation_id and "result" in json_obj and "responseId" in json_obj["result"]:
                        parent_response_id = json_obj["result"].get("parentResponseId")
                        if parent_response_id:
                            conversation_id = parent_response_id.split('-')[0]
                            logger.info(f"Extracted conversation ID from response: {conversation_id}")
                            yield {
                                "type": "conversation_id",
                                "id": conversation_id
                            }
                            
                except json.JSONDecodeError:
                    logger.warning(f"Couldn't parse line as JSON: {line_str[:100]}...")
                    continue
            
            await process.wait()
            stderr = await process.stderr.read()
            stderr_str = stderr.decode()
            if process.returncode != 0:
                if "429" in stderr_str:
                    logger.warning(f"Rate limit hit on attempt {attempt + 1}. Retrying after delay...")
                    await asyncio.sleep(2 ** attempt)
                    continue
                if "401" in stderr_str or "403" in stderr_str:
                    logger.warning(f"Authentication failed on attempt {attempt + 1}. Cookies may be invalid. Retrying after refresh...")
                    success = await refresh_cookies()
                    if success:
                        cookie_string = get_cookie_string()
                        for i, arg in enumerate(curl_command):
                            if arg.startswith('cookie:'):
                                curl_command[i] = f'cookie: {cookie_string}'
                        continue
                error_msg = f"Curl command failed: {stderr_str[:200]}..."
                logger.error(error_msg)
                yield {"type": "error", "error": "Failed to communicate with Grok"}
                break
            break
        except Exception as e:
            error_msg = f"Unexpected error in process_curl_response: {str(e)}"
            logger.error(error_msg)
            yield {"type": "error", "error": str(e)}
            break

async def stream_response(response_gen, request: ChatCompletionRequest, bolt_conversation_id: str) -> AsyncGenerator[str, None]:
    response_id = f"chatcmpl-{str(uuid.uuid4())}"
    conversation_id = None
    full_message = ""
    final_message_sent = False
    error_encountered = False
    
    try:
        async for item in response_gen:
            logger.debug(f"Streaming item: {item}")
            if item["type"] == "error":
                error_encountered = True
                error_chunk = {
                    "error": {
                        "message": item["error"],
                        "type": "server_error"
                    }
                }
                logger.debug(f"Sending error chunk: {json.dumps(error_chunk)[:200]}...")
                yield f"data: {json.dumps(error_chunk)}\n\n"
                continue
                
            if item["type"] == "conversation_id":
                conversation_id = item["id"]
                continue
                
            elif item["type"] == "token":
                token = item["token"]
                full_message += token
                
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model or "grok-3",
                    "system_fingerprint": None,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": token
                        },
                        "logprobs": None,
                        "finish_reason": None
                    }]
                }
                logger.debug(f"Sending streaming chunk: {json.dumps(chunk)[:200]}...")
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.01)
                
            elif item["type"] == "final_message" and not final_message_sent:
                final_message_sent = True
                full_message = item["message"]
                response_id = item.get("response_id", response_id)
                if conversation_id and bolt_conversation_id in conversation_map:
                    conversation_map[bolt_conversation_id]["last_response_id"] = response_id
                    conversation_map[bolt_conversation_id]["message_sequence"] = conversation_map[bolt_conversation_id].get("message_sequence", 0) + 1
                    save_conversation_map()
                    logger.info(f"Updated last_response_id for {bolt_conversation_id} to {response_id}, sequence: {conversation_map[bolt_conversation_id]['message_sequence']}")
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model or "grok-3",
                    "system_fingerprint": None,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": ""
                        },
                        "logprobs": None,
                        "finish_reason": "stop"
                    }]
                }
                logger.debug(f"Sending final streaming chunk: {json.dumps(chunk)[:200]}...")
                yield f"data: {json.dumps(chunk)}\n\n"
        
        if not error_encountered and not final_message_sent and full_message:
            final_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model or "grok-3",
                "system_fingerprint": None,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": full_message
                    },
                    "logprobs": None,
                    "finish_reason": "stop"
                }]
            }
            logger.debug(f"Sending final streaming chunk (fallback): {json.dumps(final_chunk)[:200]}...")
            yield f"data: {json.dumps(final_chunk)}\n\n"
        
        logger.debug("Sending [DONE] message")
        yield "data: [DONE]\n\n"
        logger.info(f"Full streamed message for {response_id}: {full_message[:200]}...")
        
    except Exception as e:
        logger.error(f"Error in stream_response: {str(e)}")
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "server_error"
            }
        }
        logger.debug(f"Sending error chunk: {json.dumps(error_chunk)[:200]}...")
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

async def collect_full_response(response_gen):
    conversation_id = None
    final_message = ""
    error_message = None
    
    try:
        async for item in response_gen:
            logger.debug(f"Non-streaming item: {item}")
            if item["type"] == "error":
                error_message = item["error"]
                break
                
            if item["type"] == "conversation_id":
                conversation_id = item["id"]
            elif item["type"] == "final_message":
                final_message = item["message"]
                break
            elif item["type"] == "token":
                final_message += item["token"]
        
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

@app.get("/debug/conversation_map")
async def get_conversation_map():
    load_conversation_map()
    return {"map_size": len(conversation_map), "conversation_map": conversation_map}

@app.get("/debug/session_map")
async def get_session_map():
    load_session_map()
    return {"map_size": len(session_map), "session_map": session_map}

async def periodic_cookie_refresh():
    while True:
        logger.info("Running periodic cookie validity check...")
        if not await check_cookies_validity():
            logger.warning("Cookies are invalid. Refreshing cookies...")
            await refresh_cookies()
        else:
            logger.info("Cookies are still valid. No refresh needed.")
        await asyncio.sleep(300)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Checking initial cookie validity on startup...")
    if not await check_cookies_validity():
        logger.warning("Initial cookies are invalid. Attempting to refresh...")
        await refresh_cookies()
    task = asyncio.create_task(periodic_cookie_refresh())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        logger.info("Periodic cookie refresh task cancelled on shutdown")

app.lifespan = lifespan

async def process_response_generator(response_gen, bolt_conversation_id: str, grok_conversation_id: str, conversation_id_captured: bool) -> AsyncGenerator[Dict, None]:
    async for item in response_gen:
        if not conversation_id_captured and item["type"] == "conversation_id":
            grok_conversation_id_new = item["id"]
            conversation_map[bolt_conversation_id] = {
                "conversation_id": grok_conversation_id_new,
                "last_response_id": None,
                "message_sequence": 0
            }
            logger.info(f"MAP UPDATED: BoltAI {bolt_conversation_id} -> Grok {grok_conversation_id_new}, sequence: 0")
            save_conversation_map()
            yield item
            continue
        yield item

# Add a global throttle for new conversation creation
LAST_NEW_CONVERSATION_TIME = 0
NEW_CONVERSATION_COOLDOWN = 3600  # 1 hour in seconds

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, client: Request):
    global LAST_NEW_CONVERSATION_TIME
    try:
        load_conversation_map()
        load_session_map()
        logger.debug(f"Conversation map at start: {conversation_map}")
        logger.debug(f"Session map at start: {session_map}")
        
        logger.info(f"Received request with {len(request.messages)} messages")
        logger.debug(f"Full messages: {[m.get('role') + ': ' + str(m.get('content', ''))[:50] for m in request.messages]}")
        
        user_messages = [msg["content"] for msg in request.messages if msg.get("role") == "user" and "content" in msg]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found in the request")
        
        client_ip = client.client.host
        bolt_conversation_id = get_bolt_conversation_id(request, client_ip)
        logger.info(f"BoltAI conversation ID: {bolt_conversation_id}")
        
        conversation_data = conversation_map.get(bolt_conversation_id, {})
        grok_conversation_id = conversation_data.get("conversation_id") if conversation_data else None
        parent_response_id = conversation_data.get("last_response_id") if conversation_data else None
        message_sequence = conversation_data.get("message_sequence", 0) if conversation_data else 0
        logger.info(f"Found Grok conversation ID: {grok_conversation_id} for BoltAI ID: {bolt_conversation_id}, sequence: {message_sequence}, parentResponseId: {parent_response_id}")
        
        retry_attempted = False
        conversation_id_captured = False
        
        if grok_conversation_id and not retry_attempted:
            logger.info(f"Attempting to validate existing conversation {grok_conversation_id}")
            is_valid = await validate_conversation(grok_conversation_id)
            if not is_valid:
                logger.warning(f"Conversation {grok_conversation_id} is invalid, but will attempt to reuse unless critical error.")
                retry_attempted = True
            else:
                logger.info(f"Validated existing conversation {grok_conversation_id} successfully")

        # Decide whether to continue or create a new conversation
        if grok_conversation_id and not retry_attempted:
            logger.info(f"Continuing conversation for {bolt_conversation_id} with Grok ID {grok_conversation_id}")
            response_gen = continue_conversation(grok_conversation_id, request.messages, parent_response_id)
        else:
            current_time = time.time()
            if current_time - LAST_NEW_CONVERSATION_TIME < NEW_CONVERSATION_COOLDOWN:
                logger.warning(f"New conversation creation throttled. Reusing existing ID {grok_conversation_id} if available, or failing safely.")
                if grok_conversation_id:
                    response_gen = continue_conversation(grok_conversation_id, request.messages, parent_response_id)
                else:
                    logger.error(f"No existing conversation ID to reuse, and new conversation creation is throttled.")
                    raise HTTPException(status_code=429, detail="New conversation creation throttled. Please wait before retrying.")
            else:
                logger.info(f"Starting new conversation for {bolt_conversation_id} after cooldown check")
                LAST_NEW_CONVERSATION_TIME = current_time
                response_gen = create_new_conversation(request.messages)
        
        needs_retry = False
        processed_gen = process_response_generator(response_gen, bolt_conversation_id, grok_conversation_id, conversation_id_captured)
        
        # Process generator items to check for errors requiring retry
        response_items = []
        async for item in processed_gen:
            response_items.append(item)
            if item["type"] == "conversation_id":
                grok_conversation_id = item["id"]
                conversation_id_captured = True
            if item["type"] == "error" and "Conversation" in item["error"] and "not found" in item["error"] and not retry_attempted:
                logger.warning(f"Conversation {grok_conversation_id} expired or invalid during processing. Checking throttle before new convo.")
                retry_attempted = True
                current_time = time.time()
                if current_time - LAST_NEW_CONVERSATION_TIME < NEW_CONVERSATION_COOLDOWN:
                    logger.error(f"Throttle active. Cannot create new conversation yet for {bolt_conversation_id}. Failing safely.")
                    response_items.append({"type": "error", "error": "Conversation reuse failed, and new conversation creation is throttled."})
                    needs_retry = False
                    break
                else:
                    logger.info(f"Throttle passed. Creating new conversation for {bolt_conversation_id}")
                    LAST_NEW_CONVERSATION_TIME = current_time
                    needs_retry = True
                    message_sequence = 0
                    break
        
        if needs_retry:
            logger.info(f"Retrying with a new conversation for {bolt_conversation_id} after error and throttle check")
            response_gen = create_new_conversation(request.messages, is_retry=True)
            conversation_id_captured = False
            processed_gen = process_response_generator(response_gen, bolt_conversation_id, grok_conversation_id, conversation_id_captured)
            response_items = []
            async for item in processed_gen:
                response_items.append(item)
                if item["type"] == "conversation_id":
                    grok_conversation_id = item["id"]
                    conversation_id_captured = True
        
        if request.stream:
            logger.info(f"Returning streaming response for {bolt_conversation_id}")
            return StreamingResponse(
                stream_response(processed_gen if not needs_retry else processed_gen, request, bolt_conversation_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )
        else:
            logger.info(f"Collecting full response for {bolt_conversation_id}")
            response = await collect_full_response(processed_gen if not needs_retry else processed_gen)
            response_text = response.get("response", "")
            
            prompt_tokens = sum(len(m.get("content", "")) for m in request.messages if isinstance(m.get("content", ""), str)) // 4
            completion_tokens = len(response_text) // 4 if response_text else 0
            
            logger.info(f"Completed non-streaming response: {response_text[:100]}...")
            
            return {
                "id": f"chatcmpl-{response.get('id', str(uuid.uuid4()))}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model or "grok-3",
                "system_fingerprint": None,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "logprobs": None,
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
    finally:
        logger.debug(f"Conversation map at end: {conversation_map}")
        logger.debug(f"Session map at end: {session_map}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Grok (curl subprocess, .env-style cookies) API server...")
    cookie_string = get_cookie_string()
    if not cookie_string:
        logger.error("CRITICAL COOKIE ERROR: Cookie string is empty. Check .env or defaults.")
        sys.exit(1)
    uvicorn.run(app, host="0.0.0.0", port=8000)