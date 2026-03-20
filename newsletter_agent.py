import argparse
import html
import json
import os
import re
import sqlite3
import urllib.parse
import urllib.request
from datetime import datetime

from mlx_lm import generate, load

from newsletter_schema import DB_PATH, initialize_database

MODEL_PATH = "mlx-community/gemma-3-4b-it-4bit"
DEFAULT_DAYS = 7
DEFAULT_QUERIES = 4
DEFAULT_RESULTS_PER_QUERY = 3
MAX_ARTICLE_CHARS = 8000

print("Loading newsletter brain into RAM")
model, tokenizer = load(MODEL_PATH)
initialize_database()
print("Newsletter agent ready.")


def main():
    parser = argparse.ArgumentParser(description="Local newsletter research agent")
    parser.add_argument("--brief", help="Newsletter brief or topic")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--queries", type=int, default=DEFAULT_QUERIES)
    parser.add_argument("--results-per-query", type=int, default=DEFAULT_RESULTS_PER_QUERY)
    parser.add_argument("--output-dir", default="output/newsletters")
    args = parser.parse_args()

    brief = args.brief or input("Newsletter brief: ").strip()
    if not brief:
        raise ValueError("A newsletter brief is required")

    run_newsletter_pipeline(
        brief=brief,
        days=args.days,
        query_limit=args.queries,
        results_per_query=args.results_per_query,
        output_dir=args.output_dir,
    )


def run_newsletter_pipeline(brief, days, query_limit, results_per_query, output_dir):
    plan = build_research_plan(brief, days, query_limit)
    run_id = save_run(plan, brief)

    print("\nPlanning complete.")
    print(f"Title: {plan['title']}")

    collected_sources = []
    for query in plan["queries"]:
        print(f"\nSearching: {query}")
        results = search_web(query, results_per_query)
        for rank, result in enumerate(results, start=1):
            article_text = fetch_article_text(result["url"])
            if not article_text:
                continue

            source_digest = summarize_source(brief, plan, result, article_text)
            source_record = {
                "query": query,
                "rank_index": rank,
                "title": result["title"],
                "url": result["url"],
                "snippet": result["snippet"],
                "article_text": article_text,
                "source_summary": source_digest["summary"],
                "relevance_score": float(source_digest["relevance"]),
            }
            save_source(run_id, source_record)
            collected_sources.append(source_record)
            print(f"  Saved source: {result['title']}")

    if not collected_sources:
        raise RuntimeError("No usable sources were collected from the web search step")

    newsletter_markdown = compose_newsletter(brief, plan, collected_sources, days)
    output_path = write_newsletter_file(output_dir, plan["title"], newsletter_markdown)
    update_run_output_path(run_id, output_path)

    print("\nNewsletter generated.")
    print(f"Saved to: {output_path}")


def build_research_plan(brief, days, query_limit):
    prompt = f"""<start_of_turn>user
You are the planning brain for a newsletter agent.

Task:
- understand the brief
- decide what web searches are needed
- decide what sections the newsletter should have
- produce a plan for a newsletter covering the last {days} days

Brief:
"{brief}"

Return ONLY JSON in this format:
{{"title":"str","audience":"str","tone":"str","queries":["q1","q2"],"sections":["s1","s2","s3"]}}

Rules:
- choose exactly {query_limit} focused search queries
- make the sections useful for a real newsletter
- keep title concise
- do not include any text outside JSON
<end_of_turn>
<start_of_turn>model
{{"title":"""

    plan = generate_json_from_prompt(prompt, '{"title":', 220)
    queries = [str(item).strip() for item in plan.get("queries", []) if str(item).strip()]
    sections = [str(item).strip() for item in plan.get("sections", []) if str(item).strip()]
    if len(queries) < query_limit:
        raise ValueError("Model did not provide enough queries")
    if not sections:
        raise ValueError("Model did not provide newsletter sections")

    return {
        "title": str(plan.get("title", "")).strip() or "Weekly Newsletter",
        "audience": str(plan.get("audience", "")).strip() or "General readers",
        "tone": str(plan.get("tone", "")).strip() or "Clear and conversational",
        "queries": queries[:query_limit],
        "sections": sections,
    }


def search_web(query, max_results):
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
    html_text = fetch_url(url)
    matches = re.findall(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        html_text,
        re.IGNORECASE | re.DOTALL,
    )

    results = []
    seen_urls = set()
    for href, raw_title in matches:
        resolved_url = normalize_result_url(html.unescape(href))
        if not resolved_url or resolved_url in seen_urls:
            continue

        title = clean_text(raw_title)
        if not title:
            continue

        seen_urls.add(resolved_url)
        results.append(
            {
                "title": title,
                "url": resolved_url,
                "snippet": "",
            }
        )
        if len(results) >= max_results:
            break

    return results


def normalize_result_url(href):
    if href.startswith("//"):
        href = "https:" + href

    if "duckduckgo.com/l/" in href:
        parsed = urllib.parse.urlparse(href)
        params = urllib.parse.parse_qs(parsed.query)
        target = params.get("uddg", [""])[0]
        return urllib.parse.unquote(target)

    if href.startswith("http://") or href.startswith("https://"):
        return href

    return ""


def fetch_url(url):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8", errors="ignore")


def fetch_article_text(url):
    try:
        raw_html = fetch_url(url)
    except Exception:
        return ""

    text = raw_html
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<noscript.*?>.*?</noscript>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = clean_text(text)
    return text[:MAX_ARTICLE_CHARS]


def summarize_source(brief, plan, result, article_text):
    prompt = f"""<start_of_turn>user
You are summarizing one web source for a newsletter writer.

Newsletter brief:
"{brief}"

Planned audience:
"{plan['audience']}"

Planned sections:
{json.dumps(plan['sections'])}

Source title:
"{result['title']}"

Source URL:
"{result['url']}"

Source text:
\"\"\"
{article_text}
\"\"\"

Return ONLY JSON in this format:
{{"summary":"str","relevance":0.0,"key_points":["p1","p2","p3"]}}

Rules:
- keep summary factual and concise
- relevance must be between 0 and 1
- focus on information useful for the newsletter brief
- do not include any text outside JSON
<end_of_turn>
<start_of_turn>model
{{"summary":"""

    data = generate_json_from_prompt(prompt, '{"summary":', 220)
    summary = str(data.get("summary", "")).strip()
    if not summary:
        raise ValueError("Empty source summary")

    return {
        "summary": summary,
        "relevance": float(data.get("relevance", 0) or 0),
        "key_points": data.get("key_points", []),
    }


def compose_newsletter(brief, plan, collected_sources, days):
    compact_sources = []
    for index, source in enumerate(collected_sources, start=1):
        compact_sources.append(
            {
                "id": index,
                "title": source["title"],
                "url": source["url"],
                "summary": source["source_summary"],
                "relevance": source["relevance_score"],
            }
        )

    prompt = f"""<start_of_turn>user
You are the writing brain for a newsletter agent.

Write a complete markdown newsletter using only the source summaries provided.

Newsletter brief:
"{brief}"

Title:
"{plan['title']}"

Audience:
"{plan['audience']}"

Tone:
"{plan['tone']}"

Coverage window:
Last {days} days

Planned sections:
{json.dumps(plan['sections'])}

Source summaries:
{json.dumps(compact_sources, ensure_ascii=True)}

Requirements:
- write a strong headline
- include a short opening note
- organize the body around the planned sections
- make it readable and conversational
- use inline citations like [1], [2]
- end with a short closing note
- include a final Sources section listing [id]: title - url
- return markdown only
<end_of_turn>
<start_of_turn>model
"""

    return generate_text_from_prompt(prompt, 1200).strip()


def generate_json_from_prompt(prompt, prefill, max_tokens):
    response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    full_str = prefill + response.strip()
    clean_str = "".join(char for char in full_str if ord(char) >= 32)
    match = re.search(r"(\{.*\})", clean_str, re.DOTALL)
    if not match:
        raise ValueError("Model did not return valid JSON")
    return json.loads(match.group(1))


def generate_text_from_prompt(prompt, max_tokens):
    response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    return response.strip()


def save_run(plan, brief):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO newsletter_runs (brief, audience, tone, title, queries_json, sections_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            brief,
            plan["audience"],
            plan["tone"],
            plan["title"],
            json.dumps(plan["queries"]),
            json.dumps(plan["sections"]),
        ),
    )
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id


def save_source(run_id, source_record):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO source_items (
            run_id,
            query_text,
            rank_index,
            title,
            url,
            snippet,
            article_text,
            source_summary,
            relevance_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            source_record["query"],
            source_record["rank_index"],
            source_record["title"],
            source_record["url"],
            source_record["snippet"],
            source_record["article_text"],
            source_record["source_summary"],
            source_record["relevance_score"],
        ),
    )
    conn.commit()
    conn.close()


def update_run_output_path(run_id, output_path):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE newsletter_runs SET output_path = ? WHERE id = ?",
        (output_path, run_id),
    )
    conn.commit()
    conn.close()


def write_newsletter_file(output_dir, title, markdown):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{slugify(title)}.md"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(markdown + "\n")
    return output_path


def slugify(value):
    lowered = value.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return cleaned or "newsletter"


def clean_text(value):
    stripped = re.sub(r"\s+", " ", value)
    return stripped.strip()


if __name__ == "__main__":
    main()
