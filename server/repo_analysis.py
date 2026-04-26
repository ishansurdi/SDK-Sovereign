"""Optional public-repo analysis helpers for the play/demo surface."""
from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request


_GITHUB_RE = re.compile(r"https?://github\.com/([^/]+)/([^/#?]+)")
_SDK_HINTS = {
    "stripe": {"replacement": "razorpay", "category": "payments"},
    "twilio": {"replacement": "kaleyra", "category": "messaging"},
    "googlemaps": {"replacement": "mmi_sdk", "category": "maps"},
    "maps platform": {"replacement": "mmi_sdk", "category": "maps"},
}


@dataclass(frozen=True)
class GitHubSnapshot:
    """Small repo snapshot used for offline and Gemini-backed analysis."""

    owner: str
    repo: str
    html_url: str
    default_branch: str
    description: str
    readme_text: str
    top_level_files: list[str]


def analyze_github_repo(repo_url: str) -> dict[str, Any]:
    """Analyze a public GitHub repo and optionally enrich the result with Gemini."""
    snapshot = fetch_github_snapshot(repo_url)
    local_analysis = build_local_analysis(snapshot)
    gemini = generate_gemini_analysis(snapshot, local_analysis)
    return {
        "repo": {
            "owner": snapshot.owner,
            "name": snapshot.repo,
            "url": snapshot.html_url,
            "default_branch": snapshot.default_branch,
            "description": snapshot.description,
            "top_level_files": snapshot.top_level_files,
        },
        "local_analysis": local_analysis,
        "gemini": gemini,
    }


def fetch_github_snapshot(repo_url: str) -> GitHubSnapshot:
    """Fetch a minimal snapshot from the GitHub API for a public repository."""
    owner, repo = parse_github_url(repo_url)
    repo_json = _github_json(f"https://api.github.com/repos/{owner}/{repo}")
    default_branch = repo_json.get("default_branch") or "main"
    contents = _github_json(f"https://api.github.com/repos/{owner}/{repo}/contents")
    readme_text = _fetch_readme(owner, repo)
    file_names = [item.get("name", "") for item in contents if isinstance(item, dict)]
    return GitHubSnapshot(
        owner=owner,
        repo=repo,
        html_url=str(repo_json.get("html_url") or repo_url),
        default_branch=default_branch,
        description=str(repo_json.get("description") or ""),
        readme_text=readme_text,
        top_level_files=[name for name in file_names if name][:12],
    )


def parse_github_url(repo_url: str) -> tuple[str, str]:
    """Extract owner and repo name from a GitHub URL."""
    match = _GITHUB_RE.match(repo_url.strip())
    if not match:
        raise ValueError("expected a public GitHub repository URL")
    owner = match.group(1)
    repo = match.group(2)
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo


def build_local_analysis(snapshot: GitHubSnapshot) -> dict[str, Any]:
    """Generate a local sovereign recommendation even without external LLM access."""
    corpus = "\n".join(
        [snapshot.description, snapshot.readme_text, "\n".join(snapshot.top_level_files)]
    ).lower()
    detected_sdk = None
    recommended = None
    category = None
    for needle, hint in _SDK_HINTS.items():
        if needle in corpus:
            detected_sdk = needle
            recommended = hint["replacement"]
            category = hint["category"]
            break

    findings = []
    if detected_sdk and recommended:
        findings.append(
            f"Detected likely sanctioned SDK usage around '{detected_sdk}', so the sovereign replacement candidate is '{recommended}'."
        )
    else:
        findings.append(
            "No benchmark SDK keyword was detected, so only a generic repo summary is available until code ingestion is expanded."
        )

    findings.append(
        "This add-on currently inspects GitHub metadata, README content, and top-level files. Deep code crawling can be added later if needed."
    )
    return {
        "category": category,
        "detected_sdk": detected_sdk,
        "recommended_replacement": recommended,
        "findings": findings,
        "next_action": (
            f"Run the Lead/Auditor migration flow against a reduced snapshot focused on {recommended}."
            if recommended
            else "Collect target source files from the repo and rerun analysis with richer code context."
        ),
    }


def generate_gemini_analysis(snapshot: GitHubSnapshot, local_analysis: dict[str, Any]) -> dict[str, Any]:
    """Call Gemini when configured; otherwise return a disabled status."""
    api_key = os.environ.get("GEMINI_API")
    if not api_key:
        return {
            "enabled": False,
            "status": "missing_api",
            "model": None,
            "summary": None,
        }

    prompt = (
        "You are reviewing a public GitHub repository for a sovereign SDK migration assistant. "
        "Summarize what the repo appears to do, what integration risk stands out, and what migration steps are most plausible. "
        "Keep the answer concise and structured for an engineering dashboard.\n\n"
        f"Repository: {snapshot.html_url}\n"
        f"Description: {snapshot.description or '(none)'}\n"
        f"Top-level files: {', '.join(snapshot.top_level_files) or '(none)'}\n"
        f"README excerpt:\n{snapshot.readme_text[:6000]}\n\n"
        f"Local sovereign hint: {json.dumps(local_analysis)}"
    )
    body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ]
    }
    encoded_key = parse.quote(api_key, safe="")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={encoded_key}"
    )
    payload = json.dumps(body).encode("utf-8")
    request_obj = request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(request_obj, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        summary, status, code, retry_after_seconds = _summarize_gemini_error(detail)
        return {
            "enabled": True,
            "status": status,
            "model": "gemini-2.0-flash",
            "summary": summary,
            "code": code,
            "retry_after_seconds": retry_after_seconds,
        }
    except error.URLError as exc:
        return {
            "enabled": True,
            "status": "network_error",
            "model": "gemini-2.0-flash",
            "summary": str(exc.reason),
        }

    summary = _extract_gemini_text(data)
    return {
        "enabled": True,
        "status": "ok" if summary else "empty_response",
        "model": "gemini-2.0-flash",
        "summary": summary,
    }


def _fetch_readme(owner: str, repo: str) -> str:
    """Best-effort fetch of the repository README via the GitHub contents API."""
    try:
        data = _github_json(f"https://api.github.com/repos/{owner}/{repo}/readme")
    except error.HTTPError:
        return ""
    content = data.get("content")
    if not isinstance(content, str):
        return ""
    decoded = base64.b64decode(content.encode("utf-8"))
    return decoded.decode("utf-8", errors="replace")


def _github_json(url: str) -> Any:
    """Fetch JSON from the GitHub API with a minimal user agent."""
    request_obj = request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "sdk-sovereign-demo",
        },
    )
    with request.urlopen(request_obj, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_gemini_text(payload: dict[str, Any]) -> str | None:
    """Pull the first text segment out of a Gemini generateContent payload."""
    candidates = payload.get("candidates") or []
    for candidate in candidates:
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
        merged = "\n".join(text for text in texts if text).strip()
        if merged:
            return merged
    return None


def _summarize_gemini_error(detail: str) -> tuple[str, str, int | None, float | None]:
    """Collapse verbose Gemini API errors into compact UI-friendly status text."""
    normalized = re.sub(r"\s+", " ", detail).strip()
    try:
        payload = json.loads(detail)
    except json.JSONDecodeError:
        return normalized[:280] or "Gemini request failed.", "http_error", None, None

    error_payload = payload.get("error") or {}
    status = str(error_payload.get("status") or "http_error").lower()
    code = error_payload.get("code")
    message = re.sub(r"\s+", " ", str(error_payload.get("message") or "")).strip()
    retry_match = re.search(r"Please retry in ([0-9.]+)s", message)
    retry_after_seconds = float(retry_match.group(1)) if retry_match else None

    if status == "resource_exhausted" or "quota exceeded" in message.lower():
        summary = "Gemini quota exceeded for the configured GEMINI_API secret. Check billing/quota or retry later."
        if retry_after_seconds is not None:
            summary += f" Suggested retry: about {retry_after_seconds:.0f}s."
        return summary, status, code if isinstance(code, int) else None, retry_after_seconds

    if message:
        trimmed = message.split("For more information", 1)[0].strip()
        return trimmed[:280], status, code if isinstance(code, int) else None, retry_after_seconds

    return normalized[:280] or "Gemini request failed.", status, code if isinstance(code, int) else None, retry_after_seconds