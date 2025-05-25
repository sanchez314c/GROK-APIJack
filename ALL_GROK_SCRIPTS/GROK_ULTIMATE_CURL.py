#!/usr/bin/env python3

"""
Grok3 Ultimate API Proxy Server - Curl-based Version
Author: Consolidated from Jason Paul Michaels' work
Purpose: A FastAPI-based proxy to interface with Grok's web platform using web session credentials,
         bypassing API credit limits for unlimited usage in development environments.
Version: Ultimate - Combined best features from all recovered script versions
Approach: Uses curl subprocess with comprehensive error handling and automatic cookie refresh
"""

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

# Auto-detect and install dependencies
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

log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grok_ultimate_curl.log")
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

# Paths and environment setup
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
    sso_rw = os.getenv('GROK_SSO_RW')
    anon_userid = os.getenv('GROK_ANON_USERID')
    challenge = os.getenv('GROK_CHALLENGE')
    x_signature = os.getenv('GROK_X_SIGNATURE')

    required_env_vars = {
        'GROK_CF_CLEARANCE': cf_clearance,
        'GROK_SSO': sso,
        'GROK_ANON_USERID': anon_userid,
        'GROK_CHALLENGE': challenge,
        'GROK_X_SIGNATURE': x_signature
    }

    missing_vars = [name for name, val in required_env_vars.items() if not val]
    if missing_vars:
        logger.error(f"CRITICAL: Missing essential cookie environment variables: {', '.join(missing_vars)}")
        return ""

    cookie_parts = [
        f"x-anonuserid={anon_userid}",
        f"x-challenge={challenge}",
        f"x-signature={x_signature}",
        f"sso={sso}",
        f"cf_clearance={cf_clearance}"
    ]
    if sso_rw:
        cookie_parts.append(f"sso-rw={sso_rw}")

    final_cookie_string = "; ".join(cookie_parts)
    logger.debug(f"Constructed cookie string: {final_cookie_string[:100]}...")
    return final_cookie_string

async def check_cookies_validity(attempt_refresh_on_fail: bool = True) -> bool:
    logger.debug(f"Enter check_cookies_validity (attempt_refresh_on_fail={attempt_refresh_on_fail})")
    cookie_string = get_cookie_string()

    if not cookie_string:
        logger.warning("Cookie string is empty in check_cookies_validity.")
        if attempt_refresh_on_fail:
            logger.info("Attempting to refresh cookies due to empty string...")
            if await refresh_cookies():
                logger.info("Refresh successful after empty string. Re-validating cookies (no further auto-refresh).")
                return await check_cookies_validity(attempt_refresh_on_fail=False)
            else:
                logger.error("Refresh attempt failed when cookie string was empty.")
                return False
        else:
            logger.debug("Not attempting refresh for empty cookie string (flag is False).")
            return False

    curl_command = [
        'curl', '-f', 'https://grok.com/rest/app-chat/conversations',
        '-H', 'accept: */*',
        '-H', 'accept-language: en-GB,en;q=0.9',
        '-H', 'content-type: application/json',
        '-H', f'cookie: {cookie_string}',
        '-H', 'origin: https://grok.com',
        '-H', 'priority: u=1, i',
        '-H', 'referer: https://grok.com/',
        '-H', 'sec-ch-ua: "Not/A)Brand";v="8", "Chromium";v="126", "Brave";v="126"',
        '-H', 'sec-ch-ua-mobile: ?0',
        '-H', 'sec-ch-ua-platform: "macOS"',
        '-H', 'sec-fetch-dest: empty',
        '-H', 'sec-fetch-mode: cors',
        '-H', 'sec-fetch-site: same-origin',
        '-H', 'sec-gpc: 1',
        '-H', 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *curl_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        response_text = stdout.decode().strip()
        stderr_text = stderr.decode().strip()

        if process.returncode != 0:
            logger.warning(f"Cookie validity check (curl to /conversations) failed with code {process.returncode}. Stderr: {stderr_text[:300]}")
            if attempt_refresh_on_fail and ("401" in stderr_text or "403" in stderr_text or "forbidden" in stderr_text.lower()):
                logger.info("Attempting to refresh cookies due to curl validation failure (40x)...")
                if await refresh_cookies():
                    logger.info("Refresh successful after curl failure. Re-validating cookies (no further auto-refresh).")
                    return await check_cookies_validity(attempt_refresh_on_fail=False)
                else:
                    logger.error("Refresh attempt failed after curl validation failure.")
                    return False
            else:
                logger.debug(f"Not attempting refresh or curl error was not auth-related. Curl exit: {process.returncode}")
                return False
        try:
            data = json.loads(response_text)
            if "code" in data and data["code"] in [5, 401, 403]:
                logger.warning(f"Cookies valid by curl, but API returned error code {data['code']} for /conversations.")
                if attempt_refresh_on_fail:
                    logger.info(f"Attempting refresh due to API error code {data['code']} on /conversations endpoint...")
                    if await refresh_cookies():
                        logger.info("Refresh successful. Re-validating (no further auto-refresh).")
                        return await check_cookies_validity(attempt_refresh_on_fail=False)
                    else:
                        logger.error(f"Refresh failed after API error code {data['code']}.")
                        return False
                return False
            logger.info("Cookies appear valid: curl to /conversations OK and no disqualifying API codes.")
            return True
        except json.JSONDecodeError:
            logger.error(f"Failed to parse /conversations response as JSON: {response_text[:200]}")
            return False
    except Exception as e:
        logger.error(f"Exception during cookie validity check: {str(e)}", exc_info=True)
        return False

async def refresh_cookies() -> bool:
    logger.info("Attempting to refresh cookies via Playwright using Google SSO...")
    if not GOOGLE_EMAIL or not GOOGLE_PASSWORD:
        logger.error("Google credentials not found in .env. Cannot refresh cookies.")
        return False
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
            )
            page = await context.new_page()
            logger.info("Navigating to GrokWWW...")
            await page.goto('https://grok.com', timeout=60000)
            logger.info('Clicking "Sign in with X"...')
            await page.click('text=Sign in with X', timeout=30000)
            await page.wait_for_load_state("networkidle", timeout=60000)
            logger.info("Attempting to log in via Google SSO...")
            await page.click('button[data-testid="OAuth_Provider_Google"]', timeout=30000)
            await page.wait_for_load_state("networkidle", timeout=60000)
            current_url = page.url
            if "accounts.google.com" in current_url:
                if "identifier" in current_url or "ServiceLogin" in current_url :
                    logger.info("Entering Google email...")
                    await page.fill('input[type="email"]', GOOGLE_EMAIL, timeout=30000)
                    await page.click('button:has-text("Next")', timeout=30000)
                    await page.wait_for_load_state("networkidle", timeout=60000)
                    logger.info("Entering Google password...")
                    password_input_selector = 'input[type="password"]'
                    await page.wait_for_selector(password_input_selector, timeout=30000)
                    await page.fill(password_input_selector, GOOGLE_PASSWORD, timeout=30000)
                    await page.click('button:has-text("Next")', timeout=30000)
                    await page.wait_for_load_state("networkidle", timeout=60000)
                else:
                    logger.info("Google account selection page detected. Selecting account...")
                    await page.click(f'div[data-identifier="{GOOGLE_EMAIL}"]', timeout=30000)
                    await page.wait_for_load_state("networkidle", timeout=60000)
            logger.info("Waiting for potential redirect to X.com...")
            try:
                await page.wait_for_url('https://x.com/**', timeout=30000)
                logger.info("Successfully authenticated with Google SSO, redirected to X...")
                await page.wait_for_load_state("networkidle", timeout=60000)
            except Exception as e:
                logger.warning(f"Did not redirect to x.com as expected or timed out, current URL: {page.url}. Error: {e}")
                if "grok.com" not in page.url:
                    logger.error(f"SSO flow failed, current URL: {page.url}")
                    try:
                        html_content = await page.content()
                        logger.debug(f"Page content at failure: {html_content[:500]}")
                    except Exception as pe:
                        logger.error(f"Could not get page content: {pe}")
                    await browser.close()
                    return False
            logger.info("Waiting for final redirect to GrokWWW...")
            await page.wait_for_url('https://grok.com/**', timeout=60000)
            await page.wait_for_load_state("networkidle", timeout=60000)
            logger.info("Extracting cookies from grok.com...")
            cookies = await context.cookies('https://grok.com')
            if not cookies:
                logger.error("No cookies found for grok.com after login.")
                await browser.close()
                return False
            cookie_map = {cookie['name']: cookie['value'] for cookie in cookies}
            
            env_cookie_map = {
                'GROK_CF_CLEARANCE': cookie_map.get('cf_clearance', ''),
                'GROK_SSO': cookie_map.get('sso', ''),
                'GROK_SSO_RW': cookie_map.get('sso-rw', ''),
                'GROK_ANON_USERID': cookie_map.get('x-anonuserid', ''),
                'GROK_CHALLENGE': cookie_map.get('x-challenge', ''),
                'GROK_X_SIGNATURE': cookie_map.get('x-signature', '')
            }
            missing_cookies = [key for key in ['GROK_CF_CLEARANCE', 'GROK_SSO', 'GROK_ANON_USERID', 'GROK_CHALLENGE', 'GROK_X_SIGNATURE'] if not env_cookie_map[key]]
            if missing_cookies:
                logger.error(f"Missing essential cookies after refresh: {missing_cookies}. Available: {list(cookie_map.keys())}")
                await browser.close()
                return False

            logger.info("Updating .env file with new cookies...")
            for key, value in env_cookie_map.items():
                if value:
                    set_key(ENV_PATH, key, value)
                    os.environ[key] = value
                elif key in os.environ:
                    del os.environ[key]

            load_dotenv(dotenv_path=ENV_PATH, override=True)
            logger.info("Cookies refreshed and .env updated successfully!")
            await browser.close()
            return True
    except Exception as e:
        logger.error(f"Failed to refresh cookies: {e}", exc_info=True)
        if 'browser' in locals() and browser.is_connected():
            await browser.close()
        return False

async def validate_conversation(conversation_id: str) -> bool:
    logger.info(f"Validating conversation ID: {conversation_id}")
    cookie_string = get_cookie_string()
    if not cookie_string:
        logger.warning("Cookie string empty before validating conversation. This is unexpected.")
        return False

    curl_command = [
        'curl', '-f', f'https://grok.com/rest/app-chat/conversations/{conversation_id}',
        '-H', 'accept: */*',
        '-H', 'accept-language: en-GB,en;q=0.9',
        '-H', 'content-type: application/json',
        '-H', f'cookie: {cookie_string}',
        '-H', 'origin: https://grok.com',
        '-H', 'priority: u=1, i',
        '-H', 'referer: https://grok.com/',
        '-H', 'sec-ch-ua: "Not/A)Brand";v="8", "Chromium";v="126", "Brave";v="126"',
        '-H', 'sec-ch-ua-mobile: ?0',
        '-H', 'sec-ch-ua-platform: "macOS"',
        '-H', 'sec-fetch-dest: empty',
        '-H', 'sec-fetch-mode: cors',
        '-H', 'sec-fetch-site: same-origin',
        '-H', 'sec-gpc: 1',
        '-H', 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    ]
    process = await asyncio.create_subprocess_exec(
        *curl_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    response = stdout.decode().strip()
    stderr_text = stderr.decode().strip()
    logger.debug(f"Validation response for {conversation_id}: {response[:200]}...")
    if stderr_text: logger.debug(f"Validation stderr for {conversation_id}: {stderr_text[:200]}...")

    if process.returncode != 0:
        logger.warning(f"Conversation {conversation_id} validation failed (curl code {process.returncode}): {stderr_text}")
        if "401" in stderr_text or "403" in stderr_text or "404" in stderr_text or "conversation not found" in response.lower() or (response and "code\":5" in response):
            logger.warning(f"Conversation {conversation_id} is not valid or accessible (40x error or API error code 5).")
            return False
        return False
    try:
        data = json.loads(response)
        if "code" in data and data["code"] == 5:
            logger.warning(f"Conversation {conversation_id} is not valid (API error code 5: Not Found).")
            return False
        logger.info(f"Conversation {conversation_id} appears valid.")
        return True
    except json.JSONDecodeError:
        logger.error(f"Failed to parse validation response for {conversation_id}: {response[:200]}...")
        return False

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
    first_user_message = None
    for msg in request.messages:
        if msg.get("role") == "user" and msg.get("content"):
            first_user_message = msg.get("content")
            break
    if first_user_message:
        clean_text = first_user_message.strip().lower()[:200]
        message_hash = hashlib.md5(clean_text.encode('utf-8')).hexdigest()
        bolt_id = f"bolt-ip-{client_ip}-{message_hash[:16]}"
        session_map[client_ip] = bolt_id
        save_session_map()
        logger.info(f"Generated new session ID for IP {client_ip} based on first message: {bolt_id}")
        return bolt_id
    system_content = ""
    for msg in request.messages:
        if msg.get("role") == "system" and msg.get("content"):
            system_content += msg.get("content", "")[:500]
    if system_content:
        clean_system = system_content.strip().lower()
        system_hash = hashlib.md5(clean_system.encode('utf-8')).hexdigest()
        bolt_id = f"bolt-sys-{system_hash[:16]}"
        logger.info(f"Generated conversation ID from system content (no user/session match): {bolt_id}")
        return bolt_id
    if request.user:
        user_hash = hashlib.md5(request.user.encode('utf-8')).hexdigest()
        bolt_id = f"bolt-user-{user_hash[:16]}"
        logger.info(f"Generated conversation ID from non-bolt user ID: {bolt_id}")
        return bolt_id
    random_id = f"bolt-rand-{str(uuid.uuid4())[:16]}"
    logger.info(f"Generated random conversation ID as fallback: {random_id}")
    return random_id

async def create_new_conversation(request_data: ChatCompletionRequest, is_retry: bool = False) -> AsyncGenerator[Dict, None]:
    messages = request_data.messages
    filtered_messages = []
    for msg in messages:
        if msg.get("role") == "system" and "json" in msg.get("content", "").lower() and "instruction" in msg.get("content", "").lower():
            logger.info("Filtering out system prompt containing 'json instruction'")
            continue
        filtered_messages.append(msg)

    system_prompt = None
    user_messages_content = []
    for message_item in filtered_messages:
        if message_item.get("role") == "system" and "content" in message_item:
            system_prompt = message_item["content"]
        elif message_item.get("role") == "user" and "content" in message_item:
            user_messages_content.append(message_item["content"])

    if not user_messages_content:
        logger.error("No user message content found for new conversation.")
        yield {"type": "error", "error": "No user message content found"}
        return

    formatted_message = user_messages_content[-1]
    if system_prompt and not is_retry:
        if not formatted_message.strip().lower().startswith("system:") and \
           not system_prompt.strip().lower() in formatted_message.strip().lower() :
            formatted_message = f"SYSTEM: {system_prompt}\n\nUSER: {formatted_message}"
    logger.info(f"Creating new conversation with formatted message (is_retry={is_retry}): {formatted_message[:100]}...")

    current_cookie_string = get_cookie_string()
    if not current_cookie_string:
        logger.error("CRITICAL: Cookie string is empty at create_new_conversation. Upstream check failed.")
        yield {"type": "error", "error": "Internal error: Missing session cookies."}
        return

    curl_command = [
        'curl', '-f', 'https://grok.com/rest/app-chat/conversations/new',
        '-H', 'accept: */*',
        '-H', 'accept-language: en-GB,en;q=0.9',
        '-H', 'content-type: application/json',
        '-H', f'cookie: {current_cookie_string}',
        '-H', 'dnt: 1',
        '-H', 'origin: https://grok.com',
        '-H', 'priority: u=1, i',
        '-H', 'referer: https://grok.com/',
        '-H', 'sec-ch-ua: "Not/A)Brand";v="8", "Chromium";v="126", "Brave";v="126"',
        '-H', 'sec-ch-ua-mobile: ?0',
        '-H', 'sec-ch-ua-platform: "macOS"',
        '-H', 'sec-fetch-dest: empty',
        '-H', 'sec-fetch-mode: cors',
        '-H', 'sec-fetch-site: same-origin',
        '-H', 'sec-gpc: 1',
        '-H', 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        '--data-raw', json.dumps({
            "temporary": False,
            "modelName": request_data.model or "grok-3",
            "message": formatted_message,
            "fileAttachments": [],
            "imageAttachments": [],
            "disableSearch": False,
            "enableImageGeneration": True,
            "returnImageBytes": False,
            "returnRawGrokInXaiRequest": False,
            "enableImageStreaming": True,
            "imageGenerationCount": 2,
            "forceConcise": False,
            "toolOverrides": {},
            "enableSideBySide": True,
            "isPreset": False,
            "sendFinalMetadata": True,
            "customInstructions": "",
            "deepsearchPreset": "",
            "isReasoning": False,
            "webpageUrls": [],
            "disableTextFollowUps": True
        })
    ]
    async for item in process_curl_response(curl_command):
        yield item

async def continue_conversation(grok_conversation_id: str, request_data: ChatCompletionRequest, parent_response_id: Optional[str] = None) -> AsyncGenerator[Dict, None]:
    messages = request_data.messages
    user_messages_content = []
    for message_item in messages:
        if message_item.get("role") == "user" and "content" in message_item:
            user_messages_content.append(message_item["content"])

    if not user_messages_content:
        logger.error(f"No user message content found for continuing conversation {grok_conversation_id}.")
        yield {"type": "error", "error": "No user message content found for continuation"}
        return

    formatted_message = user_messages_content[-1]
    logger.info(f"Continuing conversation {grok_conversation_id} with message: {formatted_message[:100]}... Parent Response ID: {parent_response_id}")

    current_cookie_string = get_cookie_string()
    if not current_cookie_string:
        logger.error(f"CRITICAL: Cookie string is empty at continue_conversation for GrokConvID {grok_conversation_id}.")
        yield {"type": "error", "error": "Internal error: Missing session cookies."}
        return

    payload = {
        "message": formatted_message,
        "parentResponseId": parent_response_id,
        "fileAttachments": [],
        "imageAttachments": [],
        "disableSearch": False,
        "enableImageGeneration": True,
        "returnImageBytes": False,
        "returnRawGrokInXaiRequest": False,
        "enableImageStreaming": True,
        "imageGenerationCount": 2,
        "forceConcise": False,
        "toolOverrides": {},
        "enableSideBySide": True,
        "isPreset": False,
        "sendFinalMetadata": True,
        "customInstructions": "",
        "deepsearchPreset": "",
        "isReasoning": False,
        "webpageUrls": [],
        "disableTextFollowUps": True
    }
    if not parent_response_id:
        logger.warning(f"No parentResponseId available for continuing conversation {grok_conversation_id}.")
        del payload["parentResponseId"]

    curl_command = [
        'curl', '-f', f'https://grok.com/rest/app-chat/conversations/{grok_conversation_id}/messages',
        '-H', 'accept: */*',
        '-H', 'accept-language: en-GB,en;q=0.9',
        '-H', 'content-type: application/json',
        '-H', f'cookie: {current_cookie_string}',
        '-H', 'dnt: 1',
        '-H', 'origin: https://grok.com',
        '-H', 'priority: u=1, i',
        '-H', f'referer: https://grok.com/chat/{grok_conversation_id}',
        '-H', 'sec-ch-ua: "Not/A)Brand";v="8", "Chromium";v="126", "Brave";v="126"',
        '-H', 'sec-ch-ua-mobile: ?0',
        '-H', 'sec-ch-ua-platform: "macOS"',
        '-H', 'sec-fetch-dest: empty',
        '-H', 'sec-fetch-mode: cors',
        '-H', 'sec-fetch-site: same-origin',
        '-H', 'sec-gpc: 1',
        '-H', 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        '--data-raw', json.dumps(payload)
    ]
    async for item in process_curl_response(curl_command, grok_conversation_id):
        yield item

async def process_curl_response(curl_command: List[str], conversation_id_input: Optional[str] = None, attempt_number: int = 1) -> AsyncGenerator[Dict, None]:
    try:
        logger.debug(f"Executing curl command (attempt {attempt_number}): {' '.join(curl_command)[:300]}...")
        process = await asyncio.create_subprocess_exec(
            *curl_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        accumulated_message = ""
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            line_str = line.decode().strip()
            if not line_str:
                continue
            logger.debug(f"Grok response line: {line_str[:200]}...")
            try:
                json_obj = json.loads(line_str)
                if "code" in json_obj and "message" in json_obj:
                    if json_obj["code"] == 5 and "Not Found" in json_obj["message"]:
                        error_msg = f"Grok API error: Conversation {conversation_id_input or 'new'} not found (Code 5)."
                        logger.error(error_msg)
                        yield {"type": "error", "error": error_msg, "grok_api_error_code": 5}

                if "result" in json_obj and "conversation" in json_obj["result"]:
                    conv_data = json_obj["result"]["conversation"]
                    if "conversationId" in conv_data:
                        new_conv_id = conv_data["conversationId"]
                        logger.info(f"Extracted new Grok conversation ID: {new_conv_id}")
                        yield {"type": "conversation_id", "id": new_conv_id}
                if "result" in json_obj and "response" in json_obj["result"] and "token" in json_obj["result"]["response"]:
                    token = json_obj["result"]["response"]["token"]
                    if token:
                        accumulated_message += token
                        yield {"type": "token", "token": token, "accumulated_message": accumulated_message}
                if "result" in json_obj and "response" in json_obj["result"] and "modelResponse" in json_obj["result"]["response"]:
                    model_response = json_obj["result"]["response"]["modelResponse"]
                    if "message" in model_response:
                        final_message_text = model_response["message"]
                        grok_msg_id = model_response.get("responseId")
                        if grok_msg_id:
                            logger.info(f"Extracted Grok message response ID: {grok_msg_id}")
                        logger.info(f"Received final message block: {final_message_text[:100]}...")
                        yield {
                            "type": "final_message",
                            "message": final_message_text,
                            "response_id": grok_msg_id or str(uuid.uuid4())
                        }
                if not conversation_id_input and "result" in json_obj and "responseId" in json_obj["result"]:
                    parent_response_id = json_obj["result"].get("parentResponseId")
                    if parent_response_id and '-' in parent_response_id:
                        potential_conv_id = parent_response_id.split('-')[0]
                        if len(potential_conv_id) > 20 and all(c in "0123456789abcdef-" for c in potential_conv_id):
                             logger.info(f"Extracted potential new Grok conversation ID (from parentResponseId): {parent_response_id}")
                             yield {"type": "conversation_id", "id": parent_response_id}
            except json.JSONDecodeError:
                logger.warning(f"Non-JSON line from Grok: {line_str[:100]}...")
                if "<html>" in line_str.lower() or "forbidden" in line_str.lower() or "error" in line_str.lower():
                    logger.error(f"Detected HTML/error in Grok's stdout stream: {line_str[:200]}")
        await process.wait()
        stderr_output = (await process.stderr.read()).decode().strip()
        if process.returncode != 0:
            logger.error(f"Curl command failed with exit code {process.returncode}. Stderr: {stderr_output[:500]}")
            if "401" in stderr_output or "403" in stderr_output or "forbidden" in stderr_output.lower():
                logger.warning("Authentication/Authorization error (401/403) from Grok. Cookies may be invalid or request blocked.")
                yield {"type": "error", "error": "Grok API returned 401/403 Forbidden. Cookies might be invalid or request blocked.", "status_code": process.returncode}
            elif "429" in stderr_output or "rate limit" in stderr_output.lower():
                logger.warning("Rate limit hit with Grok API.")
                yield {"type": "error", "error": "Grok API rate limit exceeded.", "status_code": process.returncode}
            elif "Could not resolve host" in stderr_output:
                 logger.error("Network error: Could not resolve Grok host.")
                 yield {"type": "error", "error": "Network error: Could not resolve Grok host.", "status_code": process.returncode}
            else:
                yield {"type": "error", "error": f"Failed to communicate with Grok (curl error {process.returncode}). Details: {stderr_output[:200]}", "status_code": process.returncode}
        else:
            logger.info(f"Curl command completed successfully (code 0). Stderr (if any): {stderr_output[:200]}")
    except Exception as e:
        logger.error(f"Unexpected error in process_curl_response: {str(e)}", exc_info=True)
        yield {"type": "error", "error": f"Unexpected error processing Grok response: {str(e)}"}

async def stream_response(response_gen: AsyncGenerator[Dict, None], request_data: ChatCompletionRequest, bolt_conversation_id: str) -> AsyncGenerator[str, None]:
    openai_stream_response_id = f"chatcmpl-{str(uuid.uuid4())}"
    grok_actual_conversation_id = None
    full_response_content = ""
    final_message_event_received = False
    error_yielded = False
    try:
        async for item in response_gen:
            logger.debug(f"Streaming item for BoltID {bolt_conversation_id}: {item}")
            if item.get("type") == "error":
                error_yielded = True
                error_message = item.get("error", "Unknown server error")
                grok_api_code = item.get("grok_api_error_code")
                openai_error_type = "server_error"
                if "401" in error_message or "403" in error_message or grok_api_code == 5 :
                    openai_error_type = "invalid_request_error"
                error_chunk_data = {
                    "error": { "message": error_message, "type": openai_error_type, "param": None, "code": str(grok_api_code) if grok_api_code else None }
                }
                logger.error(f"Yielding error chunk for BoltID {bolt_conversation_id}: {error_chunk_data}")
                yield f"data: {json.dumps(error_chunk_data)}\n\n"
                continue
            if item.get("type") == "conversation_id":
                grok_actual_conversation_id = item["id"]
                continue
            elif item.get("type") == "token":
                token_content = item["token"]
                full_response_content += token_content
                chunk = {
                    "id": openai_stream_response_id, "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": request_data.model or "grok-3", "system_fingerprint": None,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": token_content}, "logprobs": None, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.001)
            elif item.get("type") == "final_message":
                final_message_event_received = True
                full_response_content = item["message"]
                grok_message_response_id = item.get("response_id")
                if grok_actual_conversation_id and bolt_conversation_id in conversation_map:
                    if grok_message_response_id:
                        conversation_map[bolt_conversation_id]["last_response_id"] = grok_message_response_id
                    conversation_map[bolt_conversation_id]["message_sequence"] = conversation_map[bolt_conversation_id].get("message_sequence", -1) + 1
                    save_conversation_map()
                    logger.info(f"STREAM: Updated map for BoltID {bolt_conversation_id} (GrokConvID {grok_actual_conversation_id}): LastRespID {grok_message_response_id}, Seq {conversation_map[bolt_conversation_id]['message_sequence']}")
                final_delta_content = ""
                chunk = {
                    "id": openai_stream_response_id, "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": request_data.model or "grok-3", "system_fingerprint": None,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": final_delta_content}, "logprobs": None, "finish_reason": "stop"}]
                }
                logger.debug(f"Sending final streaming chunk (finish_reason=stop) for BoltID {bolt_conversation_id}")
                yield f"data: {json.dumps(chunk)}\n\n"
        if not error_yielded and full_response_content and not final_message_event_received:
            logger.info(f"No explicit final_message event for BoltID {bolt_conversation_id}, but content was streamed. Sending final 'stop' chunk.")
            if grok_actual_conversation_id and bolt_conversation_id in conversation_map:
                conversation_map[bolt_conversation_id]["message_sequence"] = conversation_map[bolt_conversation_id].get("message_sequence", -1) + 1
                save_conversation_map()
                logger.info(f"STREAM (fallback): Updated map for BoltID {bolt_conversation_id} (GrokConvID {grok_actual_conversation_id}): Seq {conversation_map[bolt_conversation_id]['message_sequence']}")
            chunk = {
                "id": openai_stream_response_id, "object": "chat.completion.chunk", "created": int(time.time()),
                "model": request_data.model or "grok-3", "system_fingerprint": None,
                "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
    except Exception as e:
        logger.error(f"Error during stream_response for BoltID {bolt_conversation_id}: {str(e)}", exc_info=True)
        if not error_yielded:
            error_chunk_data = {"error": {"message": f"Streaming failed: {str(e)}", "type": "server_error"}}
            yield f"data: {json.dumps(error_chunk_data)}\n\n"
    finally:
        if not error_yielded or (error_yielded and full_response_content):
            logger.debug(f"Sending [DONE] for BoltID {bolt_conversation_id}")
            yield "data: [DONE]\n\n"
        logger.info(f"Stream ended for BoltID {bolt_conversation_id}. Full response preview: {full_response_content[:100]}...")

async def collect_full_response(response_gen: AsyncGenerator[Dict, None], bolt_conversation_id: str):
    grok_actual_conversation_id = None
    final_message_text = ""
    grok_message_response_id = None
    error_detail = None
    accumulated_from_tokens = ""
    try:
        async for item in response_gen:
            logger.debug(f"Collecting item for BoltID {bolt_conversation_id}: {item}")
            if item.get("type") == "error":
                error_detail = item.get("error", "Unknown error from Grok")
                logger.error(f"Error collecting response for BoltID {bolt_conversation_id}: {error_detail}")
                break
            if item.get("type") == "conversation_id":
                grok_actual_conversation_id = item["id"]
            elif item.get("type") == "token":
                accumulated_from_tokens += item["token"]
            elif item.get("type") == "final_message":
                final_message_text = item["message"]
                grok_message_response_id = item.get("response_id")
        if error_detail:
            return {"error": error_detail, "grok_conversation_id": grok_actual_conversation_id}
        if not final_message_text and accumulated_from_tokens:
            final_message_text = accumulated_from_tokens
            logger.info(f"Collected full response for BoltID {bolt_conversation_id} from tokens as no final_message event provided explicit text.")
        logger.info(f"Collected full response for BoltID {bolt_conversation_id}, GrokConvID {grok_actual_conversation_id}. Preview: {final_message_text[:100]}...")
        return {
            "grok_conversation_id": grok_actual_conversation_id,
            "response_text": final_message_text,
            "grok_message_response_id": grok_message_response_id
        }
    except Exception as e:
        logger.error(f"Unexpected error in collect_full_response for BoltID {bolt_conversation_id}: {str(e)}", exc_info=True)
        return {"error": f"Failed to collect response: {str(e)}", "grok_conversation_id": grok_actual_conversation_id}

@app.get("/debug/conversation_map")
async def get_conversation_map_debug():
    load_conversation_map()
    return {"map_size": len(conversation_map), "conversation_map": conversation_map}

@app.get("/debug/session_map")
async def get_session_map_debug():
    load_session_map()
    return {"map_size": len(session_map), "session_map": session_map}

async def periodic_cookie_refresh():
    while True:
        await asyncio.sleep(1800)
        logger.info("Running periodic cookie validity check and refresh...")
        if not await check_cookies_validity():
            logger.warning("Periodic check found cookies invalid. `check_cookies_validity` should have attempted a refresh.")
            logger.error("Periodic cookie refresh attempt FAILED or cookies remain invalid.")
        else:
            logger.info("Periodic check: Cookies are still valid.")

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    logger.info("Application startup: Checking initial cookie validity...")
    if not await check_cookies_validity():
        logger.warning("Initial cookies are invalid or check failed. `check_cookies_validity` should attempt refresh.")
    else:
        logger.info("Initial cookie check passed.")
    refresh_task = asyncio.create_task(periodic_cookie_refresh())
    logger.info("Periodic cookie refresh task started.")
    yield
    logger.info("Application shutdown: Cancelling periodic cookie refresh task...")
    refresh_task.cancel()
    try:
        await refresh_task
    except asyncio.CancelledError:
        logger.info("Periodic cookie refresh task successfully cancelled.")
    except Exception as e:
        logger.error(f"Error during shutdown of periodic refresh task: {e}", exc_info=True)
    logger.info("Application shutdown complete.")

app.router.lifespan_context = lifespan

async def process_response_generator_wrapper(
    response_gen: AsyncGenerator[Dict, None],
    bolt_conversation_id: str,
    initial_grok_conv_id: Optional[str]
) -> AsyncGenerator[Dict, None]:
    grok_id_already_processed_for_this_stream = False
    async for item in response_gen:
        if item.get("type") == "conversation_id" and not grok_id_already_processed_for_this_stream:
            new_grok_conv_id = item["id"]
            if new_grok_conv_id and new_grok_conv_id != initial_grok_conv_id:
                conversation_map[bolt_conversation_id] = {
                    "conversation_id": new_grok_conv_id,
                    "last_response_id": None,
                    "message_sequence": 0
                }
                save_conversation_map()
                logger.info(f"MAP NEW/UPDATE for BoltID {bolt_conversation_id}: New GrokConvID {new_grok_conv_id}, Seq reset to 0.")
                grok_id_already_processed_for_this_stream = True
        yield item

@app.post("/v1/chat/completions")
async def chat_completions(request_data: ChatCompletionRequest, http_request: Request):
    bolt_conversation_id = None
    try:
        load_conversation_map()
        load_session_map()
        client_ip = http_request.client.host if http_request.client else "unknown_ip"
        bolt_conversation_id = get_bolt_conversation_id(request_data, client_ip)
        logger.info(f"Request for BoltID {bolt_conversation_id} (IP: {client_ip}). Model: {request_data.model}, Stream: {request_data.stream}")
        logger.debug(f"BoltID {bolt_conversation_id} - Request messages: {[str(m.get('content', ''))[:60] for m in request_data.messages]}")
        if not any(msg.get("role") == "user" and msg.get("content") for msg in request_data.messages):
            logger.error(f"BoltID {bolt_conversation_id} - No user message found in request.")
            raise HTTPException(status_code=400, detail="No user message found in the request.")

        max_retries = 2
        current_attempt = 0
        grok_response_generator = None
        initial_grok_data = conversation_map.get(bolt_conversation_id, {})
        grok_conv_id_from_map = initial_grok_data.get("conversation_id")
        last_resp_id_from_map = initial_grok_data.get("last_response_id")
        final_processed_stream = None

        while current_attempt < max_retries:
            current_attempt += 1
            logger.info(f"BoltID {bolt_conversation_id} - Attempt {current_attempt}/{max_retries}")
            if not await check_cookies_validity(attempt_refresh_on_fail=True):
                logger.error(f"BoltID {bolt_conversation_id} - Cookie validation and internal refresh attempts failed (Attempt {current_attempt}).")
                if current_attempt >= max_retries:
                    logger.error(f"BoltID {bolt_conversation_id} - Cookies persistently invalid after {max_retries} attempts.")
                    raise HTTPException(status_code=503, detail="Service unavailable: Failed to establish secure session.")
                await asyncio.sleep(2)
                grok_conv_id_from_map = None
                last_resp_id_from_map = None
                continue
            
            use_existing_grok_conversation = False
            if grok_conv_id_from_map:
                logger.info(f"BoltID {bolt_conversation_id} - Found GrokConvID {grok_conv_id_from_map} in map. Validating...")
                if await validate_conversation(grok_conv_id_from_map):
                    logger.info(f"BoltID {bolt_conversation_id} - GrokConvID {grok_conv_id_from_map} is VALID. Will continue.")
                    use_existing_grok_conversation = True
                else:
                    logger.warning(f"BoltID {bolt_conversation_id} - GrokConvID {grok_conv_id_from_map} is INVALID. Will start new.")
                    if bolt_conversation_id in conversation_map: del conversation_map[bolt_conversation_id]
                    save_conversation_map()
                    grok_conv_id_from_map = None
                    last_resp_id_from_map = None
            
            if use_existing_grok_conversation:
                logger.info(f"BoltID {bolt_conversation_id} - Continuing Grok conversation {grok_conv_id_from_map} (ParentRespID: {last_resp_id_from_map})")
                grok_response_generator = continue_conversation(grok_conv_id_from_map, request_data, last_resp_id_from_map)
            else:
                logger.info(f"BoltID {bolt_conversation_id} - Starting NEW Grok conversation.")
                grok_response_generator = create_new_conversation(request_data, is_retry=(current_attempt > 1))
                grok_conv_id_from_map = None
                last_resp_id_from_map = None
            
            processed_grok_stream = process_response_generator_wrapper(grok_response_generator, bolt_conversation_id, grok_conv_id_from_map)
            
            first_item = None
            try:
                first_item = await anext(processed_grok_stream)
            except StopAsyncIteration:
                logger.error(f"BoltID {bolt_conversation_id} - Grok response generator was empty.")
                if current_attempt < max_retries: await asyncio.sleep(1); continue
                else: raise HTTPException(status_code=502, detail="Failed to get response from upstream service (empty generator).")

            if first_item and first_item.get("type") == "error":
                error_msg = first_item.get("error", "Unknown error from Grok.")
                grok_status_code = first_item.get("status_code")
                logger.warning(f"BoltID {bolt_conversation_id} - Immediate error from Grok data stream: {error_msg} (Curl Status: {grok_status_code})")
                if ("401" in error_msg or "403" in error_msg or "Forbidden" in error_msg) and current_attempt < max_retries:
                    logger.info(f"BoltID {bolt_conversation_id} - Auth error from Grok DATA endpoint. Will retry, forcing re-check of cookies.")
                    grok_conv_id_from_map = None; last_resp_id_from_map = None
                    await asyncio.sleep(1); continue
                elif current_attempt >= max_retries:
                    logger.error(f"BoltID {bolt_conversation_id} - Max retries reached. Last error from Grok data stream: {error_msg}")
                    raise HTTPException(status_code=502, detail=f"Upstream service error after retries: {error_msg}")
                else:
                    logger.info(f"BoltID {bolt_conversation_id} - Non-auth error from Grok data stream, will retry. Error: {error_msg}")
                    await asyncio.sleep(1); continue
            
            async def combined_stream_gen():
                if first_item: yield first_item
                async for item_ in processed_grok_stream: yield item_
            
            final_processed_stream = combined_stream_gen()
            break
        else:
            logger.error(f"BoltID {bolt_conversation_id} - Failed to get valid Grok response after {max_retries} attempts.")
            raise HTTPException(status_code=503, detail="Service unavailable after multiple attempts.")

        if request_data.stream:
            logger.info(f"BoltID {bolt_conversation_id} - Returning StreamingResponse.")
            return StreamingResponse(
                stream_response(final_processed_stream, request_data, bolt_conversation_id),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            logger.info(f"BoltID {bolt_conversation_id} - Collecting full response (non-streaming).")
            collected_data = await collect_full_response(final_processed_stream, bolt_conversation_id)
            if "error" in collected_data:
                logger.error(f"BoltID {bolt_conversation_id} - Error in collected non-streaming response: {collected_data['error']}")
                raise HTTPException(status_code=502, detail=f"Upstream service error: {collected_data['error']}")
            response_text = collected_data.get("response_text", "")
            final_grok_conv_id = collected_data.get("grok_conversation_id")
            final_grok_msg_id = collected_data.get("grok_message_response_id")
            if final_grok_conv_id and bolt_conversation_id in conversation_map:
                conv_entry = conversation_map[bolt_conversation_id]
                conv_entry["conversation_id"] = final_grok_conv_id
                if final_grok_msg_id: conv_entry["last_response_id"] = final_grok_msg_id
                conv_entry["message_sequence"] = conv_entry.get("message_sequence", -1) + 1
                save_conversation_map()
                logger.info(f"NON-STREAM: Updated map for BoltID {bolt_conversation_id} (GrokConvID {final_grok_conv_id}): LastRespID {final_grok_msg_id}, Seq {conv_entry['message_sequence']}")
            elif final_grok_conv_id :
                if bolt_conversation_id not in conversation_map:
                    conversation_map[bolt_conversation_id] = {
                        "conversation_id": final_grok_conv_id, "last_response_id": final_grok_msg_id, "message_sequence": 0
                    }
                    save_conversation_map()
                    logger.info(f"NON-STREAM: Created new map entry for BoltID {bolt_conversation_id} (GrokConvID {final_grok_conv_id}): LastRespID {final_grok_msg_id}, Seq 0")
                elif final_grok_msg_id:
                     conversation_map[bolt_conversation_id]["last_response_id"] = final_grok_msg_id
                     save_conversation_map()
            prompt_tokens = sum(len(str(m.get("content", ""))) for m in request_data.messages) // 4
            completion_tokens = len(response_text) // 4
            openai_response_id = f"chatcmpl-{final_grok_conv_id or str(uuid.uuid4())}"
            return {
                "id": openai_response_id, "object": "chat.completion", "created": int(time.time()),
                "model": request_data.model or "grok-3", "system_fingerprint": None,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "logprobs": None, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled error in chat_completions for BoltID {bolt_conversation_id or 'UNKNOWN'}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        logger.debug(f"Request completed for BoltID {bolt_conversation_id or 'UNKNOWN'}.")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Grok Ultimate Curl-based API server...")
    initial_cookie_string = get_cookie_string()
    if not initial_cookie_string:
        logger.warning("CRITICAL COOKIE WARNING: Initial cookie string is empty. Check .env. Lifespan will attempt validation/refresh.")
    else:
        logger.info("Initial .env cookies loaded. Lifespan will validate.")
    uvicorn.run(app, host="0.0.0.0", port=8000)