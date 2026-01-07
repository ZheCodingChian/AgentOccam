import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from playwright.sync_api import Page, Request, Response


class NetworkLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_file = None
        self.action_counter = 0
        self.request_counter = 0
        self.response_counter = 0

        self.content_types_whitelist = [
            'application/json',
            'text/html',
            'text/plain',
            'application/xml',
            'text/xml',
            'application/javascript',
            'text/javascript',
        ]

        self.domain_blacklist = [
            'google-analytics',
            'googletagmanager',
            'doubleclick',
            'facebook.net',
            'analytics',
            'tracking',
            'telemetry',
            'cdn.segment.com',
            'mixpanel.com',
        ]

        self.media_types_blacklist = [
            'image/',
            'video/',
            'audio/',
            'font/',
        ]

        self.pending_requests = {}

        self._open_log_file()

    def _open_log_file(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        self._write_header()

    def _write_header(self):
        header = f"""
{'='*100}
Network Activity Log
Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*100}

"""
        self.log_file.write(header)
        self.log_file.flush()

    def attach_to_page(self, page: Page):
        page.on("request", self._on_request)
        page.on("response", self._on_response)

    def log_action(self, action: dict[str, Any], element_info: dict[str, str]):
        self.action_counter += 1
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        action_type = action.get('action_type', 'unknown')
        action_str = self._format_action_string(action)

        element_detail = ""
        if element_info:
            role = element_info.get('role', '')
            name = element_info.get('name', '')
            element_id = element_info.get('id', '')
            if role and name:
                element_detail = f"\nElement: {role} \"{name}\" [ID: {element_id}]"
            elif element_id:
                element_detail = f"\nElement: [ID: {element_id}]"

        log_entry = f"""
{'='*100}
[{timestamp}] ACTION #{self.action_counter}
Action: {action_str}{element_detail}
{'='*100}

"""
        self.log_file.write(log_entry)
        self.log_file.flush()

    def _format_action_string(self, action: dict[str, Any]) -> str:
        action_type = action.get('action_type', 'unknown')

        from browser_env.actions import ActionTypes

        if action_type == ActionTypes.CLICK:
            if action.get('element_id'):
                return f"click [{action['element_id']}]"
            return "click"
        elif action_type == ActionTypes.TYPE:
            element_id = action.get('element_id', '')
            text = action.get('text', [])
            if isinstance(text, list):
                from browser_env.actions import _id2key
                text_str = ''.join([_id2key.get(idx, '') for idx in text])
            else:
                text_str = str(text)
            return f"type [{element_id}] [{text_str}]"
        elif action_type == ActionTypes.SCROLL:
            direction = action.get('direction', '')
            return f"scroll {direction}"
        elif action_type == ActionTypes.GOTO_URL:
            url = action.get('url', '')
            return f"goto {url}"
        elif action_type == ActionTypes.HOVER:
            element_id = action.get('element_id', '')
            return f"hover [{element_id}]"
        elif action_type == ActionTypes.STOP:
            return "stop"
        else:
            return f"action_type_{action_type}"

    def _on_request(self, request: Request):
        if not self._should_log_request(request):
            return

        self.request_counter += 1
        request_id = request.url + str(time.time())

        self.pending_requests[request_id] = {
            'request_num': self.request_counter,
            'request': request,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        }

        self._log_request(request, self.request_counter, self.pending_requests[request_id]['timestamp'])

    def _on_response(self, response: Response):
        request = response.request
        request_id = request.url + str(time.time())

        if not self._should_log_response(response):
            return

        self.response_counter += 1

        pending_req = None
        for rid, req_data in self.pending_requests.items():
            if req_data['request'].url == request.url:
                pending_req = req_data
                break

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        self._log_response(response, self.response_counter, timestamp, pending_req)

    def _should_log_request(self, request: Request) -> bool:
        url = request.url.lower()

        for domain in self.domain_blacklist:
            if domain in url:
                return False

        resource_type = request.resource_type
        if resource_type in ['image', 'media', 'font', 'stylesheet', 'manifest', 'other']:
            return False

        return True

    def _should_log_response(self, response: Response) -> bool:
        url = response.url.lower()

        for domain in self.domain_blacklist:
            if domain in url:
                return False

        content_type = response.headers.get('content-type', '').lower()

        for media_type in self.media_types_blacklist:
            if content_type.startswith(media_type):
                return False

        should_whitelist = False
        for allowed_type in self.content_types_whitelist:
            if allowed_type in content_type:
                should_whitelist = True
                break

        if content_type and not should_whitelist:
            return False

        return True

    def _log_request(self, request: Request, request_num: int, timestamp: str):
        method = request.method
        url = request.url
        headers = dict(request.headers)

        page_context = ""
        try:
            if request.frame and request.frame.page:
                page = request.frame.page
                page_title = page.title()
                page_url = page.url
                page_context = f"Page: \"{page_title}\" ({page_url})\n"
        except Exception:
            pass

        post_data = ""
        try:
            if request.post_data:
                post_data = request.post_data
                if len(post_data) > 500:
                    post_data = f"{post_data[:500]}... [TRUNCATED - Total size: {len(request.post_data)} chars]"
        except Exception:
            post_data = "[Unable to retrieve post data]"

        log_entry = f"""[{timestamp}] NETWORK REQUEST #{request_num}
{page_context}Method: {method}
URL: {url}
Headers: {json.dumps(headers, indent=2)}"""

        if post_data:
            log_entry += f"\nBody: {post_data}"

        log_entry += f"\n{'-'*100}\n\n"

        self.log_file.write(log_entry)
        self.log_file.flush()

    def _log_response(self, response: Response, response_num: int, timestamp: str, pending_req: dict | None):
        status = response.status
        status_text = response.status_text
        url = response.url
        headers = dict(response.headers)

        page_context = ""
        try:
            if response.frame and response.frame.page:
                page = response.frame.page
                page_title = page.title()
                page_url = page.url
                page_context = f"Page: \"{page_title}\" ({page_url})\n"
        except Exception:
            pass

        body = ""
        try:
            body_text = response.text()
            if body_text:
                if len(body_text) > 500:
                    body = f"{body_text[:500]}... [TRUNCATED - Total size: {len(body_text)} chars]"
                else:
                    body = body_text
        except Exception as e:
            body = f"[Unable to retrieve response body: {str(e)}]"

        request_num_str = ""
        if pending_req:
            request_num_str = f" (Request #{pending_req['request_num']})"

        log_entry = f"""[{timestamp}] NETWORK RESPONSE #{response_num}{request_num_str}
{page_context}Status: {status} {status_text}
URL: {url}
Headers: {json.dumps(headers, indent=2)}"""

        if body:
            log_entry += f"\nBody: {body}"

        log_entry += f"\n{'='*100}\n\n"

        self.log_file.write(log_entry)
        self.log_file.flush()

    def close(self):
        if self.log_file:
            footer = f"""
{'='*100}
Network Activity Log Ended
Total Actions: {self.action_counter}
Total Requests: {self.request_counter}
Total Responses: {self.response_counter}
Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*100}
"""
            self.log_file.write(footer)
            self.log_file.close()
            self.log_file = None
