import json
import os
import re
from typing import List, Dict, Tuple
import requests
from rapidfuzz import fuzz


def extract_stop_answer(trajectory_path: str) -> str:
    """Extract the answer from the 'stop' action in trajectory.json"""
    with open(trajectory_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    trajectory = data.get('trajectory', [])

    for step in reversed(trajectory):
        action = step.get('action', '')
        if action.startswith('stop ['):
            answer = action[6:-1]
            return answer

    raise ValueError("No 'stop' action found in trajectory")


def format_with_llm(answer: str) -> tuple[Dict, str]:
    """Send answer to DeepSeek via OpenRouter to extract titles"""

    prompt = f"""You are a data extraction assistant. Given the following text containing news headlines or information, extract ALL distinct titles/headlines as exact strings (do not paraphrase or modify).

Input text:
{answer}

Requirements:
1. Extract each distinct title/headline exactly as written
2. Do not modify, paraphrase, or summarize the titles
3. Preserve exact punctuation, capitalization, and wording
4. Return a JSON object with a "titles" array

Output format:
{{
  "titles": ["exact title 1", "exact title 2", ...]
}}

Return ONLY the JSON object, no other text."""

    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek/deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )

    response.raise_for_status()
    result = response.json()

    llm_output = result['choices'][0]['message']['content']

    json_match = re.search(r'\{[\s\S]*\}', llm_output)
    if json_match:
        formatted_data = json.loads(json_match.group())
    else:
        formatted_data = json.loads(llm_output)

    return formatted_data, llm_output


def parse_network_log(network_log_path: str) -> List[Dict]:
    """Parse network_log.txt into structured format with full response objects"""
    with open(network_log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    entries = []

    response_pattern = r'\[.*?\] NETWORK RESPONSE #\d+[^\n]*\n([\s\S]*?)(?=\n\[.*?\] NETWORK RESPONSE #|\n={70,}|\Z)'
    matches = re.finditer(response_pattern, content)

    for match in matches:
        block = match.group(1)

        url_match = re.search(r'URL: (.+?)(?:\n|$)', block)
        status_match = re.search(r'Status: (.+?)(?:\n|$)', block)
        headers_match = re.search(r'Headers: (\{[\s\S]*?\n\})', block)
        body_match = re.search(r'Body: ([\s\S]*?)$', block)

        if url_match:
            url = url_match.group(1).strip()
            status = status_match.group(1).strip() if status_match else ""
            headers = headers_match.group(1).strip() if headers_match else ""
            body = body_match.group(1).strip() if body_match else ""

            entries.append({
                "url": url,
                "status": status,
                "headers": headers,
                "body": body,
                "response_body": body
            })

    return entries


def match_title_to_responses(title: str, network_entries: List[Dict]) -> List[Dict]:
    """Match single title against all responses, return matching entries"""
    matched_entries = []
    for entry in network_entries:
        if title in entry["response_body"]:
            matched_entries.append(entry)
    return matched_entries


def fuzzy_match_title_to_responses(
    title: str,
    network_entries: List[Dict],
    threshold: int = 80
) -> List[Tuple[Dict, float]]:
    """Fuzzy match a single title against all responses using token_set_ratio"""
    matched_entries = []
    for entry in network_entries:
        score = fuzz.token_set_ratio(title, entry["response_body"])
        if score >= threshold:
            matched_entries.append((entry, score))

    matched_entries.sort(key=lambda x: x[1], reverse=True)
    return matched_entries


def match_all_titles(
    titles: List[str],
    network_entries: List[Dict],
    fuzzy_threshold: int = 80
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Tuple[Dict, float]]]]:
    """Two-tier matching: exact then fuzzy. Returns (exact_matches, fuzzy_matches)"""
    exact_matches = {}
    unmatched_titles = []

    for title in titles:
        matched_entries = match_title_to_responses(title, network_entries)
        if matched_entries:
            exact_matches[title] = matched_entries
        else:
            unmatched_titles.append(title)

    fuzzy_matches = {}
    for title in unmatched_titles:
        matches = fuzzy_match_title_to_responses(title, network_entries, threshold=fuzzy_threshold)
        if matches:
            fuzzy_matches[title] = matches

    return exact_matches, fuzzy_matches


def main():
    trajectory_path = "/Users/zhe/Devtools/Github/AgentOccam/output/test10/260107_170439/trajectory.json"
    network_log_path = "/Users/zhe/Devtools/Github/AgentOccam/output/test10/260107_170439/network_log.txt"

    print(f"Trajectory: {trajectory_path}. \nNetwork log: {network_log_path}\n")

    answer = extract_stop_answer(trajectory_path)
    formatted_data, llm_response = format_with_llm(answer)

    print(f"LLM response: {llm_response}\n")
    print(f"Extracted {len(formatted_data['titles'])} titles\n")

    network_entries = parse_network_log(network_log_path)
    print(f"Parsing {len(network_entries)} network responses\n")

    exact_matches, fuzzy_matches = match_all_titles(formatted_data["titles"], network_entries, fuzzy_threshold=80)

    total_titles = len(formatted_data["titles"])
    exact_count = len(exact_matches)
    fuzzy_count = len(fuzzy_matches)
    no_match_count = total_titles - exact_count - fuzzy_count

    print("Results:")
    print(f"{exact_count} titles exact string match found")
    print(f"{fuzzy_count} titles fuzzy string match found")
    print(f"{no_match_count} titles no match found\n")

    if exact_matches:
        print("Exact Matches:\n")
        for title, entries in exact_matches.items():
            print(f"Title: {title}")
            for entry in entries:
                print(f"Network response:")
                print(f"  URL: {entry['url']}")
                print(f"  Status: {entry['status']}")
                print(f"  Headers: {entry['headers']}")
                print(f"  Body: {entry['body'][:500]}...")
                print()

    if fuzzy_matches:
        print("Fuzzy Matches:\n")
        for title, matches in fuzzy_matches.items():
            print(f"Title: {title}")
            for entry, score in matches:
                print(f"Score: {score:.1f}%")
                print(f"Network response:")
                print(f"  URL: {entry['url']}")
                print(f"  Status: {entry['status']}")
                print(f"  Headers: {entry['headers']}")
                print(f"  Body: {entry['body'][:500]}...")
                print()

    if no_match_count > 0:
        print("Titles with no matches:")
        all_matched_titles = set(exact_matches.keys()) | set(fuzzy_matches.keys())
        for title in formatted_data["titles"]:
            if title not in all_matched_titles:
                print(f"  - {title}")


if __name__ == "__main__":
    main()
