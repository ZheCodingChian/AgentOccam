"""Implements helper functions to assist evaluation cases where other evaluators are not suitable."""
from typing import Any

from playwright.sync_api import Page
from llms.providers.openai_utils import (
    generate_from_openai_chat_completion,
)


def llm_fuzzy_match(pred: str, reference: str, question: str) -> float:
    """Check whether the prediction matches the reference with GPT4-turbo"""
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = "Help a teacher to grade the answer of a student given a question. Keep in mind that the student has performed the action to get the answer. They are allowed to use different phrasing or wording to answer the question. The goal is to evaluate whether the key points in the reference answer are included in the student's answer. We allow answers with additional information that doesn't contradict the reference answer and review them as fully (not partially) correct.\n"
    message += f"question: {question}\n"
    message += f"reference answer: {reference}\n"
    message += "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
    message += f"student answer: {pred}\n"
    message += "Conclude the judgement by correct/incorrect/partially correct and explain why."
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]
    response = generate_from_openai_chat_completion(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0,
        max_tokens=768,
        top_p=1.0,
        context_length=0,
    ).lower()
    print(response)
    if "partially correct" in response or "incorrect" in response:
        return 0.0
    else:
        assert "correct" in response
        return 1.0

def llm_ua_match(pred: str, reference: str, question: str) -> float:
    """Check whether the prediction matches the reference with GPT-turbo"""
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = ""
    message += f"task: {question}\n"
    message += f"actual unachievable reason: {reference}\n"
    message += f"reported unachievable reason: {pred}\n"
    message += (
        "The task described above is inherently unachievable due to the reason specified under 'actual unachievable reason'. "
        "An individual previously attempted this task and was unable to complete it. They provided a reason for their failure, "
        "which is listed under 'reported unachievable reason'. Your role is to review both the actual and reported reasons. "
        "Determine if the reported reason aligns with the actual reason, even if implicitly. "
        "If the stated reason is in line with the actual reason, respond with 'same'. Otherwise, respond with 'different'."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]

    response = generate_from_openai_chat_completion(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0,
        max_tokens=768,
        top_p=1.0,
        context_length=0,
    ).lower()
    if "different" in response:
        return 0.0
    else:
        assert "same" in response
        return 1.0


class PseudoPage:
    def __init__(self, original_page: Page, url: str):
        self.url = url
        self.original_page = original_page

    def __getattr__(self, attr: str) -> Any:
        # Delegate attribute access to the original page object
        if attr not in ["url"]:
            return getattr(self.original_page, attr)
        else:
            return getattr(self, attr)
