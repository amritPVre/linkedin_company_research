# app.py — Streamlit LinkedIn Company Finder using OpenRouter + DeepSeek v3.1:online
# Run:  streamlit run app.py

import json
import re
import io
import os
from typing import List, Dict

import string
import pandas as pd
import requests
import streamlit as st

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="LinkedIn Company Finder", page_icon="🔎", layout="wide")

st.title("🔎 LinkedIn Company Finder — DeepSeek v3.1:online")
st.caption(
    "Search public LinkedIn company pages by keyword, location, size & starting letter. "
    "Powered by OpenRouter Web Search + DeepSeek v3.1."
)

with st.expander("⚠️ Notes & Compliance", expanded=False):
    st.markdown(
        """
        - This tool searches the open web and **returns links to public LinkedIn *company* pages**.
        - It does **not** log in or scrape behind authentication, and avoids personal profiles.
        - API keys are **loaded from Streamlit Secrets** or environment variables — no key field shown in the UI.
        - Respect LinkedIn terms of service and local regulations. Verify results before outreach.
        """
    )

# -----------------------------
# API & Model Settings (NO UI KEY FIELD)
# -----------------------------
# Priority 1: Streamlit secrets; Priority 2: Environment variable
API_KEY = st.secrets.get("OPENROUTER_API_KEY", "") or os.getenv("OPENROUTER_API_KEY", "")
MODEL = "deepseek/deepseek-chat-v3.1:online"  # ':online' enables OpenRouter web search
MAX_COMPANIES_DEFAULT = 50
ENABLE_REASONING = True  # toggle here if desired

if not API_KEY:
    st.error(
        "OpenRouter API key not found. Add it to Streamlit Secrets as `OPENROUTER_API_KEY` "
        "(or set the environment variable `OPENROUTER_API_KEY`).\n\n"
        "**Local dev:** create `.streamlit/secrets.toml` with:\n\n"
        "```toml\nOPENROUTER_API_KEY = \"sk-or-...\"\n```"
    )

HEADERS = {
    "Authorization": f"Bearer {API_KEY}" if API_KEY else "",
    "HTTP-Referer": "http://localhost:8501",  # adjust in production
    "X-Title": "LinkedIn Company Finder",
    "Content-Type": "application/json",
}

# -----------------------------
# UI — Search Form
# -----------------------------
with st.form("search-form", clear_on_submit=False):
    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    with c1:
        keywords = st.text_input(
            "Keywords",
            placeholder="solar pv, epc, renewable energy",
            help="Use commas to separate multiple keywords. We'll match any of them.",
        )
    with c2:
        location = st.text_input(
            "Location filter",
            placeholder="India, Oman, Middle East, USA, Bangalore, etc.",
        )
    with c3:
        size = st.selectbox(
            "Company size",
            [
                "any",
                "1-10",
                "11-50",
                "51-200",
                "201-500",
                "501-1,000",
                "1,001-5,000",
                "5,001-10,000",
                "10,001+",
            ],
            index=0,
            help="Matches LinkedIn's public size tags where available.",
        )
    with c4:
        letter_options = ["All letters (A–Z)"] + list(string.ascii_uppercase)
        starts_with_choice = st.selectbox(
            "Starts with",
            options=letter_options,
            index=0,
            help="Pick a specific letter or choose 'All letters (A–Z)' to disable the filter.",
        )
        starts_with = "" if starts_with_choice == "All letters (A–Z)" else starts_with_choice

    max_companies = st.slider("Max companies", 1, 200, MAX_COMPANIES_DEFAULT, step=1)
    submitted = st.form_submit_button("Search Companies 🚀")

# -----------------------------
# Helpers
# -----------------------------
JSON_REGEX = re.compile(r"\[.*\]", re.DOTALL)


def build_user_prompt(keywords: str, location: str, size: str, starts_with: str, k: int) -> str:
    return f"""
You are a web research assistant.
Task: Find up to {k} **public LinkedIn company pages** that match these filters.

Filters:
- Keywords (match any): {keywords or 'N/A'}
- Location (city/region/country words must appear on the page or metadata): {location or 'N/A'}
- Company size (use LinkedIn size tag when visible): {size}
- Company name starts with: {starts_with or 'any'}

Rules:
1) Use web search. Consider reputable sources and prioritize LinkedIn **company** pages only (URLs like `https://www.linkedin.com/company/...`).
2) Exclude personal profiles, job posts, posts, or sales pages.
3) Return **ONLY** a compact JSON array (no prose) with objects of shape:
   {{"name": "Company Name", "linkedin_url": "https://www.linkedin.com/company/..."}}
4) Ensure URLs are canonical company pages (avoid tracking params) and unique by company domain.
5) Prefer companies whose page text/snippet shows the keyword(s) and location signal.
6) If nothing is found, return `[]`.

Output: JSON array only.
"""


def extract_json_array(text: str) -> List[Dict]:
    if not text:
        return []
    # Try to find a JSON array substring
    m = JSON_REGEX.search(text)
    raw = m.group(0) if m else text.strip()
    try:
        data = json.loads(raw)
        # normalize
        rows = []
        for item in data:
            name = str(item.get("name", "")).strip()
            url = str(item.get("linkedin_url", "")).strip()
            if name and url and "/company/" in url:
                # remove query params/fragments
                url = url.split("?")[0].split("#")[0]
                rows.append({"name": name, "linkedin_url": url})
        # dedupe by URL
        seen = set()
        unique = []
        for r in rows:
            if r["linkedin_url"] not in seen:
                seen.add(r["linkedin_url"])
                unique.append(r)
        return unique
    except Exception:
        return []


# -----------------------------
# Action — Call OpenRouter
# -----------------------------
if submitted:
    if not API_KEY:
        st.stop()

    user_prompt = build_user_prompt(keywords, location, size, (starts_with or "").upper().strip(), max_companies)

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise research model. Use the web plugin to find public LinkedIn company pages. "
                    "Always return only valid JSON as requested."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 2000,
        "temperature": 0.2,
    }

    if ENABLE_REASONING:
        payload["reasoning"] = {"effort": "medium", "enabled": True}

    with st.spinner("Searching the web and compiling companies…"):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=HEADERS,
                data=json.dumps(payload),
                timeout=90,
            )
            if resp.status_code != 200:
                st.error(f"OpenRouter API error: {resp.status_code} — {resp.text[:500]}")
                st.stop()

            data = resp.json()
            # Extract assistant content (supports both OpenAI-like and possible variants)
            content = None
            try:
                content = data["choices"][0]["message"]["content"]
            except Exception:
                content = (
                    data.get("choices", [{}])[0]
                    .get("messages", [{}])[0]
                    .get("content", "")
                )

            rows = extract_json_array(content)
            if not rows:
                st.warning("No valid companies found. Try relaxing filters or increasing the max companies.")
                st.json(content)
                st.stop()

            df = pd.DataFrame(rows)
            st.success(f"Found {len(df)} companies.")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Downloads
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="companies.csv", mime="text/csv")

            xlsx_buf = io.BytesIO()
            with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="companies")
            st.download_button(
                "⬇️ Download Excel",
                data=xlsx_buf.getvalue(),
                file_name="companies.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # Show raw model output for transparency
            with st.expander("Model raw output"):
                st.code(content)

        except requests.exceptions.RequestException as e:
            st.error(f"Network/timeout error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    ---
    **Setup tip:** For Streamlit Cloud or local dev, put your key in `.streamlit/secrets.toml` as:\n\n
    ```toml
    OPENROUTER_API_KEY = "sk-or-..."
    ```
    """
)
