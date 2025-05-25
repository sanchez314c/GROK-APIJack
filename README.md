# 🚀 GROK3 Unofficial API Access Collection

**Get jacked w/ Grok - Two powerful approaches to access GROK3 AI without official API support**

This repository provides **two complete implementations** for accessing GROK3 programmatically, bypassing the need for manual browser interaction and enabling seamless integration into your applications.

## 📁 Repository Structure

```
grok3-APIJack/
├── grok3-APIJack/              # FastAPI Proxy Server Collection
│   ├── grok3_fastapi_proxy_minimal.py
│   ├── grok3_fastapi_proxy_playwright_autoinstall.py ⭐
│   └── requirements.txt
│
└── grok3-api/                  # Python SDK Package
    ├── grok_client/
    │   ├── __init__.py
    │   └── client.py           # Main client implementation
    ├── pyproject.toml
    └── requirements.txt
```

## 🚀 Quick Start

### FastAPI Proxy (Recommended)

```bash
git clone https://github.com/sanchez314c/grok3-APIJack.git
cd grok3-APIJack/grok3-APIJack
python grok3_fastapi_proxy_playwright_autoinstall.py
```

### Python SDK

```bash
cd grok3-APIJack/grok3-api
pip install -e .
```

```python
from grok_client import GrokClient

cookies = {"auth_token": "your_token"}
client = GrokClient(cookies)
response = client.send_message("Hello GROK3!")
```

## License

MIT License - see LICENSE file for details.