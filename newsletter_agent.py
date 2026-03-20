import argparse
import html
import json
import os
import platform
import re
import sqlite3
import ssl
import subprocess
import sys
import urllib.parse
import urllib.error
import urllib.request
from datetime import datetime

from newsletter_schema import DB_PATH, initialize_database

DEFAULT_MODEL_PATH = "mlx-community/gemma-3-4b-it-4bit"
DEFAULT_DAYS = 7
DEFAULT_QUERIES = 4
DEFAULT_RESULTS_PER_QUERY = 3
MAX_ARTICLE_CHARS = 8000
REQUEST_TIMEOUT_SECONDS = 5
DEFAULT_WRITING_STYLE = "Sharp, analytical, and premium. Write like a high-end newsletter editor, not a corporate content bot."
SLICE_MODEL_PATHS = {
    0.125: os.environ.get("NEWSLETTER_AGENT_MODEL_SLICE_12_5", DEFAULT_MODEL_PATH),
    0.25: os.environ.get("NEWSLETTER_AGENT_MODEL_SLICE_25", DEFAULT_MODEL_PATH),
    0.50: os.environ.get("NEWSLETTER_AGENT_MODEL_SLICE_50", DEFAULT_MODEL_PATH),
    0.75: os.environ.get("NEWSLETTER_AGENT_MODEL_SLICE_75", DEFAULT_MODEL_PATH),
    1.00: os.environ.get("NEWSLETTER_AGENT_MODEL_SLICE_100", DEFAULT_MODEL_PATH),
}
EXPLANATION_STYLE_GUIDANCE = {
    "concise": "Explain ideas briefly and cleanly. Prioritize compression, signal, and fast readability.",
    "feynman": "Explain ideas simply and clearly, as if teaching an intelligent beginner. Break down jargon into plain language without sounding childish.",
    "soc": "Reveal the reasoning path in a lightweight way. Show how one point leads to the next, using a reflective and exploratory style without becoming rambling.",
}
DEPTH_PRESETS = {
    "low": {
        "query_limit": 2,
        "results_per_query": 2,
        "article_chars": 4000,
        "summary_tokens": 160,
        "newsletter_tokens": 900,
    },
    "medium": {
        "query_limit": 4,
        "results_per_query": 3,
        "article_chars": 8000,
        "summary_tokens": 220,
        "newsletter_tokens": 1200,
    },
    "high": {
        "query_limit": 6,
        "results_per_query": 5,
        "article_chars": 12000,
        "summary_tokens": 320,
        "newsletter_tokens": 1700,
    },
}
MODEL_PROFILES = {
    "constrained": {
        "model_path": os.environ.get("NEWSLETTER_AGENT_MODEL_LOW", DEFAULT_MODEL_PATH),
        "draft_model_path": os.environ.get("NEWSLETTER_AGENT_DRAFT_MODEL_LOW", ""),
        "lazy": True,
        "num_draft_tokens": 0,
    },
    "balanced": {
        "model_path": os.environ.get("NEWSLETTER_AGENT_MODEL_MEDIUM", DEFAULT_MODEL_PATH),
        "draft_model_path": os.environ.get("NEWSLETTER_AGENT_DRAFT_MODEL_MEDIUM", ""),
        "lazy": False,
        "num_draft_tokens": 0,
    },
    "expanded": {
        "model_path": os.environ.get("NEWSLETTER_AGENT_MODEL_HIGH", DEFAULT_MODEL_PATH),
        "draft_model_path": os.environ.get("NEWSLETTER_AGENT_DRAFT_MODEL_HIGH", ""),
        "lazy": False,
        "num_draft_tokens": 4,
    },
}
INSECURE_SSL_CONTEXT = ssl._create_unverified_context()
initialize_database()

MLX_GENERATE = None
MLX_LOAD = None
MODEL = None
TOKENIZER = None
MODEL_RUNTIME = None
DEVICE_CLASS_OVERRIDE = os.environ.get("NEWSLETTER_AGENT_DEVICE_CLASS", "").strip().lower() or None


def main():
    parser = argparse.ArgumentParser(description="Local newsletter research agent")
    parser.add_argument("--brief", help="Newsletter brief or topic")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--depth", choices=("low", "medium", "high"))
    parser.add_argument("--device-class", choices=("macbook", "midrange_laptop", "gaming_laptop", "midrange_phone", "flagship_phone"))
    parser.add_argument("--explanation-style", choices=("concise", "feynman", "soc", "custom"))
    parser.add_argument("--style-instructions")
    parser.add_argument("--queries", type=int)
    parser.add_argument("--results-per-query", type=int)
    parser.add_argument("--output-dir", default="output/newsletters")
    args = parser.parse_args()

    brief = args.brief or input("Newsletter brief: ").strip()
    if not brief:
        raise ValueError("A newsletter brief is required")

    initialize_model_runtime(args.device_class)

    depth = args.depth or prompt_for_depth()
    explanation_style, custom_style_instructions = resolve_explanation_style(
        args.explanation_style,
        args.style_instructions,
    )
    settings = build_research_settings(depth, args.queries, args.results_per_query)

    run_newsletter_pipeline(
        brief=brief,
        days=args.days,
        depth=depth,
        explanation_style=explanation_style,
        custom_style_instructions=custom_style_instructions,
        settings=settings,
        output_dir=args.output_dir,
    )


def initialize_model_runtime(device_class_override=None):
    global MLX_GENERATE
    global MLX_LOAD
    global MODEL
    global TOKENIZER
    global MODEL_RUNTIME
    global DEVICE_CLASS_OVERRIDE

    if MODEL_RUNTIME is not None:
        return MODEL_RUNTIME

    if device_class_override:
        DEVICE_CLASS_OVERRIDE = device_class_override

    system_info = detect_system_info()
    runtime_profile = choose_model_profile(system_info, DEVICE_CLASS_OVERRIDE)

    print(f"Using model slice: {runtime_profile['slice_label']}")

    if runtime_profile["runtime_backend"] != "mlx":
        raise SystemExit(
            f"Detected {runtime_profile['device_class']} and selected matformer slice "
            f"{runtime_profile['slice_label']}. This codebase currently ships only the MLX "
            f"runtime loader, so non-Mac execution still needs a compatible backend to be added "
            f"for {runtime_profile['runtime_backend']}."
        )

    if not system_info["metal_supported"]:
        raise SystemExit("No Metal-capable Apple device detected. The MLX runtime requires Metal support.")

    try:
        from mlx_lm import generate as mlx_generate, load as mlx_load
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: mlx_lm.\n"
            "Run this script with the project virtual environment:\n"
            "  gemma-env/bin/python newsletter_agent.py --brief \"Your newsletter brief\"\n"
            "Or activate it first:\n"
            "  source gemma-env/bin/activate"
        ) from exc

    print("Loading newsletter brain into RAM")
    model, tokenizer = mlx_load(
        runtime_profile["model_path"],
        lazy=runtime_profile["lazy"],
    )

    draft_model = None
    if runtime_profile["draft_model_path"]:
        try:
            draft_model, _ = mlx_load(runtime_profile["draft_model_path"], lazy=True)
            print(f"Loaded draft model: {runtime_profile['draft_model_path']}")
        except Exception as exc:
            print(f"Draft model load failed, continuing without it: {exc}")

    MLX_GENERATE = mlx_generate
    MLX_LOAD = mlx_load
    MODEL = model
    TOKENIZER = tokenizer
    MODEL_RUNTIME = {
        "system_info": system_info,
        "device_class": runtime_profile["device_class"],
        "profile_name": runtime_profile["profile_name"],
        "runtime_backend": runtime_profile["runtime_backend"],
        "slice_ratio": runtime_profile["slice_ratio"],
        "model_path": runtime_profile["model_path"],
        "draft_model": draft_model,
        "num_draft_tokens": runtime_profile["num_draft_tokens"],
    }

    print("Newsletter agent ready.")
    return MODEL_RUNTIME


def detect_system_info():
    system_name = platform.system()
    profiler_data = run_system_profiler(system_name)
    hardware = profiler_data.get("SPHardwareDataType", [{}])
    displays = profiler_data.get("SPDisplaysDataType", [{}])
    hardware_info = hardware[0] if hardware else {}
    display_info = displays[0] if displays else {}

    gpu_info = detect_gpu_info(system_name, display_info)
    memory_total_gb = detect_total_memory_gb(system_name, hardware_info)
    memory_available_gb = detect_available_memory_gb(system_name)
    is_android = bool(os.environ.get("ANDROID_ROOT") or os.environ.get("TERMUX_VERSION"))
    is_ios = sys.platform == "ios"
    chip = hardware_info.get("chip_type", "")
    machine = platform.machine()
    is_apple_silicon = system_name == "Darwin" and (
        machine == "arm64" or str(chip).startswith("Apple ")
    )
    metal_supported = (
        display_info.get("spdisplays_metal") == "spdisplays_supported" or is_apple_silicon
    )

    return {
        "system_name": system_name,
        "platform": platform.platform(),
        "machine": machine,
        "chip": chip,
        "hardware_model": hardware_info.get("machine_model", ""),
        "gpu_model": gpu_info["model"],
        "gpu_vendor": gpu_info["vendor"],
        "gpu_memory_gb": gpu_info["memory_gb"],
        "dedicated_gpu": gpu_info["dedicated"],
        "gpu_cores": display_info.get("sppci_cores", ""),
        "metal_supported": metal_supported,
        "is_apple_silicon": is_apple_silicon,
        "is_android": is_android,
        "is_ios": is_ios,
        "is_mobile": is_android or is_ios,
        "memory_total_gb": memory_total_gb,
        "memory_available_gb": memory_available_gb,
    }


def run_system_profiler(system_name):
    if system_name != "Darwin":
        return {}

    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
            check=True,
        )
        return json.loads(result.stdout)
    except Exception:
        return {}


def detect_gpu_info(system_name, display_info):
    if system_name == "Darwin":
        return {
            "model": str(display_info.get("sppci_model", "")),
            "vendor": "apple",
            "memory_gb": 0.0,
            "dedicated": False,
        }

    nvidia_info = detect_nvidia_gpu()
    if nvidia_info:
        return nvidia_info

    pci_hint = detect_pci_gpu_hint()
    if pci_hint:
        return pci_hint

    return {
        "model": "",
        "vendor": "",
        "memory_gb": 0.0,
        "dedicated": False,
    }


def detect_nvidia_gpu():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
            check=True,
        )
    except Exception:
        return None

    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not line:
        return None

    parts = [part.strip() for part in line.split(",")]
    memory_gb = (float(parts[1]) / 1024.0) if len(parts) > 1 else 0.0
    return {
        "model": parts[0],
        "vendor": "nvidia",
        "memory_gb": memory_gb,
        "dedicated": True,
    }


def detect_pci_gpu_hint():
    try:
        result = subprocess.run(
            ["lspci"],
            capture_output=True,
            text=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
            check=True,
        )
    except Exception:
        return None

    output = result.stdout.lower()
    keywords = ("nvidia", "geforce", "rtx", "gtx", "radeon", "amd")
    if not any(keyword in output for keyword in keywords):
        return None

    return {
        "model": "discrete-gpu",
        "vendor": "nvidia/amd",
        "memory_gb": 0.0,
        "dedicated": True,
    }


def detect_total_memory_gb(system_name, hardware_info):
    memory_text = str(hardware_info.get("physical_memory", "")).strip()
    if memory_text:
        match = re.match(r"([0-9.]+)\s*GB", memory_text, re.IGNORECASE)
        if match:
            return float(match.group(1))

    if system_name in {"Linux", "Android"} or os.environ.get("ANDROID_ROOT"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        kilobytes = int(line.split()[1])
                        return kilobytes / (1024**2)
        except Exception:
            pass

    if system_name == "Windows":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            memory_status = MemoryStatus()
            memory_status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))
            return memory_status.ullTotalPhys / (1024**3)
        except Exception:
            pass

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        return (page_size * total_pages) / (1024**3)
    except Exception:
        return 0.0


def detect_available_memory_gb(system_name):
    if system_name in {"Linux", "Android"} or os.environ.get("ANDROID_ROOT"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        kilobytes = int(line.split()[1])
                        return kilobytes / (1024**2)
        except Exception:
            pass

    if system_name == "Windows":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            memory_status = MemoryStatus()
            memory_status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))
            return memory_status.ullAvailPhys / (1024**3)
        except Exception:
            pass

    try:
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
            check=True,
        )
    except Exception:
        return 0.0

    page_size_match = re.search(r"page size of (\d+) bytes", result.stdout)
    page_size = int(page_size_match.group(1)) if page_size_match else 4096
    page_counts = {}

    for line in result.stdout.splitlines():
        match = re.match(r"([^:]+):\s+([0-9]+)\.", line.strip())
        if match:
            page_counts[match.group(1)] = int(match.group(2))

    available_pages = (
        page_counts.get("Pages free", 0)
        + page_counts.get("Pages inactive", 0)
        + page_counts.get("Pages speculative", 0)
        + page_counts.get("Pages purgeable", 0)
    )
    return (available_pages * page_size) / (1024**3)


def choose_model_profile(system_info, device_class_override=None):
    device_class = detect_device_class(system_info, device_class_override)
    slice_ratio = choose_slice_ratio(device_class, system_info)
    runtime_backend = choose_runtime_backend(device_class, system_info)

    if slice_ratio <= 0.25:
        profile_name = "constrained"
    elif slice_ratio >= 0.75:
        profile_name = "expanded"
    else:
        profile_name = "balanced"

    profile = dict(MODEL_PROFILES[profile_name])
    profile["profile_name"] = profile_name
    profile["device_class"] = device_class
    profile["runtime_backend"] = runtime_backend
    profile["slice_ratio"] = slice_ratio
    profile["slice_label"] = format_slice_label(slice_ratio)
    profile["model_path"] = choose_model_path_for_slice(slice_ratio, profile["model_path"])
    return profile


def detect_device_class(system_info, device_class_override=None):
    override = device_class_override or DEVICE_CLASS_OVERRIDE
    if override:
        return override

    total_gb = system_info["memory_total_gb"]
    if system_info["is_mobile"]:
        return "flagship_phone" if total_gb >= 8 else "midrange_phone"

    if system_info["system_name"] == "Darwin":
        return "macbook"

    if system_info["dedicated_gpu"] or system_info["gpu_memory_gb"] >= 6 or total_gb >= 24:
        return "gaming_laptop"

    return "midrange_laptop"


def choose_slice_ratio(device_class, system_info):
    if device_class == "midrange_phone":
        return 0.125
    if device_class == "flagship_phone":
        return 0.25
    if device_class == "midrange_laptop":
        return 0.50
    if device_class == "macbook":
        return 0.75
    if device_class == "gaming_laptop":
        return 1.00

    return 0.25


def choose_runtime_backend(device_class, system_info):
    if device_class == "macbook" and (
        system_info["metal_supported"] or system_info.get("is_apple_silicon")
    ):
        return "mlx"
    if device_class in {"midrange_phone", "flagship_phone"}:
        return "mobile_matformer"
    if device_class == "gaming_laptop":
        return "desktop_gpu_matformer"
    return "desktop_cpu_matformer"


def choose_model_path_for_slice(slice_ratio, default_path):
    rounded_ratio = round(slice_ratio, 3)
    return SLICE_MODEL_PATHS.get(rounded_ratio, default_path)


def format_slice_label(slice_ratio):
    percentage = slice_ratio * 100
    if percentage.is_integer():
        return f"{int(percentage)}%"
    return f"{percentage:.1f}%"


def run_newsletter_pipeline(
    brief,
    days,
    depth,
    explanation_style,
    custom_style_instructions,
    settings,
    output_dir,
):
    plan = build_research_plan(brief, days, settings["query_limit"], depth)
    run_id = save_run(plan, brief, depth, explanation_style, custom_style_instructions)
    market_snapshot = fetch_market_snapshot(brief)

    print("\nPlanning complete.")
    print(f"Title: {plan['title']}")
    print(f"Research depth: {depth}")
    print(f"Explanation style: {explanation_style}")
    if market_snapshot:
        print(f"Structured market data collected for {len(market_snapshot)} assets.")

    collected_sources = []
    for query in plan["queries"]:
        print(f"\nSearching: {query}")
        results = search_web(query, settings["results_per_query"])
        if not results:
            print("  No search results collected for this query.")
            continue
        for rank, result in enumerate(results, start=1):
            article_text = fetch_article_text(result["url"], settings["article_chars"])
            if not article_text:
                continue

            try:
                source_digest = summarize_source(
                    brief,
                    plan,
                    result,
                    article_text,
                    settings["summary_tokens"],
                )
            except Exception as exc:
                print(f"  Skipped source: summary failed: {exc}")
                continue

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
        raise RuntimeError(
            "No usable sources were collected from the web search step. "
            "The search provider may be blocking requests or the network may still be failing."
        )

    newsletter_markdown = compose_newsletter(
        brief,
        plan,
        collected_sources,
        days,
        market_snapshot,
        depth,
        explanation_style,
        custom_style_instructions,
        settings["newsletter_tokens"],
    )
    output_path = write_newsletter_file(output_dir, plan["title"], newsletter_markdown)
    update_run_output_path(run_id, output_path)

    print("\nNewsletter generated.")
    print(f"Saved to: {output_path}")


def build_research_settings(depth, query_limit_override, results_per_query_override):
    settings = dict(DEPTH_PRESETS[depth])
    if query_limit_override is not None:
        settings["query_limit"] = query_limit_override
    if results_per_query_override is not None:
        settings["results_per_query"] = results_per_query_override
    return settings


def prompt_for_depth():
    while True:
        value = input("Research depth (low/medium/high) [medium]: ").strip().lower()
        if not value:
            return "medium"
        if value in DEPTH_PRESETS:
            return value
        print("Please choose low, medium, or high.")


def resolve_explanation_style(explanation_style, style_instructions):
    style = explanation_style or prompt_for_explanation_style()
    custom_instructions = ""

    if style == "custom":
        custom_instructions = (style_instructions or input("Custom style instructions: ").strip())
        if not custom_instructions:
            raise ValueError("Custom style instructions are required when explanation style is custom")
    elif style_instructions:
        custom_instructions = style_instructions.strip()

    return style, custom_instructions


def prompt_for_explanation_style():
    while True:
        value = input("Explanation style (concise/feynman/soc/custom) [concise]: ").strip().lower()
        if not value:
            return "concise"
        if value in {"concise", "feynman", "soc", "custom"}:
            return value
        print("Please choose concise, feynman, soc, or custom.")


def build_explanation_guidance(explanation_style, custom_style_instructions):
    if explanation_style == "custom":
        return custom_style_instructions.strip()
    return EXPLANATION_STYLE_GUIDANCE[explanation_style]


def build_research_plan(brief, days, query_limit, depth):
    prompt = f"""<start_of_turn>user
You are the planning brain for a newsletter agent.

Task:
- understand the brief
- decide what web searches are needed
- decide what sections the newsletter should have
- produce a plan for a newsletter covering the last {days} days
- adjust the breadth of research for {depth} depth

Brief:
"{brief}"

Return ONLY JSON in this format:
{{"title":"str","audience":"str","tone":"str","queries":["q1","q2"],"sections":["s1","s2","s3"]}}

Rules:
- choose exactly {query_limit} focused search queries
- make the sections useful for a real newsletter
- keep title concise
- for low depth, keep the scope tight
- for medium depth, balance breadth and efficiency
- for high depth, cover the topic more comprehensively
- do not include any text outside JSON
<end_of_turn>
<start_of_turn>model
{{"title":"""

    try:
        plan = generate_json_from_prompt(
            prompt,
            '{"title":',
            220,
            schema_hint='{"title":"str","audience":"str","tone":"str","queries":["q1","q2"],"sections":["s1","s2","s3"]}',
        )
    except Exception as exc:
        print(f"Planning fallback activated: {exc}")
        return build_fallback_research_plan(brief, days, query_limit, depth)

    queries = [str(item).strip() for item in plan.get("queries", []) if str(item).strip()]
    sections = [str(item).strip() for item in plan.get("sections", []) if str(item).strip()]
    if len(queries) < query_limit:
        print("Planning fallback activated: model did not provide enough queries")
        return build_fallback_research_plan(brief, days, query_limit, depth)
    if not sections:
        print("Planning fallback activated: model did not provide newsletter sections")
        return build_fallback_research_plan(brief, days, query_limit, depth)

    return {
        "title": str(plan.get("title", "")).strip() or "Weekly Newsletter",
        "audience": str(plan.get("audience", "")).strip() or "General readers",
        "tone": str(plan.get("tone", "")).strip() or DEFAULT_WRITING_STYLE,
        "queries": queries[:query_limit],
        "sections": sections,
    }


def build_fallback_research_plan(brief, days, query_limit, depth):
    normalized_brief = clean_text(brief)
    query_candidates = [
        f"{normalized_brief}",
        f"{normalized_brief} latest developments last {days} days",
        f"{normalized_brief} key news this week",
        f"{normalized_brief} analysis and outlook",
        f"{normalized_brief} expert commentary",
        f"{normalized_brief} what changed this week",
    ]

    queries = []
    seen = set()
    for query in query_candidates:
        lowered = query.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        queries.append(query)
        if len(queries) >= query_limit:
            break

    sections = [
        "What happened",
        "Why it matters",
        "Key developments",
        "What to watch next",
    ]

    return {
        "title": generate_fallback_title(normalized_brief),
        "audience": "General readers who want a useful weekly briefing",
        "tone": DEFAULT_WRITING_STYLE,
        "queries": queries,
        "sections": sections[: max(3, min(len(sections), 4 if depth == "high" else 3))],
    }


def generate_fallback_title(brief):
    words = [word for word in re.split(r"\s+", brief) if word]
    compact = " ".join(words[:4]).strip()
    if not compact:
        return "Weekly Newsletter"
    return compact.title()


def fetch_market_snapshot(brief):
    if not looks_like_crypto_brief(brief):
        return []

    url = (
        "https://api.coingecko.com/api/v3/coins/markets"
        "?vs_currency=usd&order=market_cap_desc&per_page=5&page=1"
        "&sparkline=false&price_change_percentage=7d"
    )

    try:
        payload = fetch_url(url)
        data = json.loads(payload)
    except Exception as exc:
        print(f"Structured market data fetch failed: {exc}")
        return []

    snapshot = []
    for item in data[:5]:
        snapshot.append(
            {
                "name": item.get("name", ""),
                "symbol": str(item.get("symbol", "")).upper(),
                "price_usd": item.get("current_price"),
                "market_cap_rank": item.get("market_cap_rank"),
                "change_24h_pct": item.get("price_change_percentage_24h"),
                "change_7d_pct": item.get("price_change_percentage_7d_in_currency"),
            }
        )

    return snapshot


def looks_like_crypto_brief(brief):
    lowered = brief.lower()
    keywords = (
        "crypto",
        "bitcoin",
        "ethereum",
        "blockchain",
        "defi",
        "token",
        "altcoin",
        "web3",
    )
    return any(keyword in lowered for keyword in keywords)


def search_web(query, max_results):
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
    try:
        html_text = fetch_url(url)
    except Exception as exc:
        print(f"  Search fetch failed: {exc}")
        return []
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
    return read_response(request)


def read_response(request):
    with urllib.request.urlopen(
        request,
        timeout=REQUEST_TIMEOUT_SECONDS,
        context=INSECURE_SSL_CONTEXT,
    ) as response:
        return response.read().decode("utf-8", errors="ignore")


def fetch_article_text(url, max_article_chars):
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
    return text[:max_article_chars]


def summarize_source(brief, plan, result, article_text, summary_tokens):
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

    data = generate_json_from_prompt(
        prompt,
        '{"summary":',
        summary_tokens,
        schema_hint='{"summary":"str","relevance":0.0,"key_points":["p1","p2","p3"]}',
    )
    summary = str(data.get("summary", "")).strip()
    if not summary:
        raise ValueError("Empty source summary")

    return {
        "summary": summary,
        "relevance": float(data.get("relevance", 0) or 0),
        "key_points": data.get("key_points", []),
    }


def compose_newsletter(
    brief,
    plan,
    collected_sources,
    days,
    market_snapshot,
    depth,
    explanation_style,
    custom_style_instructions,
    newsletter_tokens,
):
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

    market_data_block = json.dumps(market_snapshot, ensure_ascii=True)
    explanation_guidance = build_explanation_guidance(
        explanation_style,
        custom_style_instructions,
    )
    editorial_brief = build_editorial_brief(
        brief,
        plan,
        compact_sources,
        market_snapshot,
        depth,
    )

    prompt = f"""<start_of_turn>user
You are the writing brain for a newsletter agent.

Write a complete markdown newsletter using the structured market data, editorial brief, and source summaries provided.

Newsletter brief:
"{brief}"

Title:
"{plan['title']}"

Audience:
"{plan['audience']}"

Tone:
"{plan['tone']}"

House style:
"{DEFAULT_WRITING_STYLE}"

Explanation style:
{explanation_style}

Explanation guidance:
"{explanation_guidance}"

Coverage window:
Last {days} days

Research depth:
{depth}

Planned sections:
{json.dumps(plan['sections'])}

Editorial brief:
{json.dumps(editorial_brief, ensure_ascii=True)}

Structured market data:
{market_data_block}

Source summaries:
{json.dumps(compact_sources, ensure_ascii=True)}

Requirements:
- write a strong headline
- include a short opening note with a point of view
- organize the body around the planned sections
- make it readable, analytical, and confident
- do not sound generic, padded, or corporate
- do not merely aggregate events; synthesize them into a sharp argument
- prove the thesis aggressively instead of hinting at it
- surface the hidden pattern the average reader would miss
- make at least one concrete interpretation about what mattered most this week
- include one killer insight that makes the reader rethink the topic
- prefer sharp-smart over safe-smart
- avoid consultant language, hedging, and empty acceleration talk
- follow the explanation guidance exactly
- if depth is low, keep it crisp and selective
- if depth is medium, balance signal and concision
- if depth is high, go deeper on implications and cross-source synthesis
- use inline citations like [1], [2]
- if structured market data is present, cite it as [M1]
- if structured market data is provided, use the exact percentage moves and prices from it instead of vague descriptions
- when you mention top market movers, include the actual numbers such as 24h or 7d percentage moves
- end with a short closing note
- include a final Sources section listing [id]: title - url
- if structured market data is present, add: [M1]: CoinGecko Markets API - https://www.coingecko.com/
- return markdown only
<end_of_turn>
<start_of_turn>model
"""

    return generate_text_from_prompt(prompt, newsletter_tokens).strip()


def build_editorial_brief(brief, plan, compact_sources, market_snapshot, depth):
    prompt = f"""<start_of_turn>user
You are the editorial strategist for a premium newsletter.

Your job is to transform raw source summaries into a sharp point of view before drafting begins.

Newsletter brief:
"{brief}"

Audience:
"{plan['audience']}"

Tone:
"{plan['tone']}"

Depth:
{depth}

Planned sections:
{json.dumps(plan['sections'])}

Structured market data:
{json.dumps(market_snapshot, ensure_ascii=True)}

Source summaries:
{json.dumps(compact_sources, ensure_ascii=True)}

Return ONLY JSON in this format:
{{"core_thesis":"str","hidden_pattern":"str","killer_insight":"str","contrarian_take":"str","proof_points":["p1","p2","p3"]}}

Rules:
- the thesis must be explicit and defensible
- the hidden pattern must go beyond restating events
- the killer insight should feel memorable and surprising
- the contrarian take should sharpen the voice without becoming fake or sensational
- proof_points should be concrete claims the final newsletter should prove
- do not include any text outside JSON
<end_of_turn>
<start_of_turn>model
{{"core_thesis":"""

    try:
        return generate_json_from_prompt(
            prompt,
            '{"core_thesis":',
            220,
            schema_hint='{"core_thesis":"str","hidden_pattern":"str","killer_insight":"str","contrarian_take":"str","proof_points":["p1","p2","p3"]}',
        )
    except Exception:
        return {
            "core_thesis": "This week was not just busy; it showed a clearer strategic direction in the field.",
            "hidden_pattern": "Parallel progress across multiple fronts usually matters more than any single headline.",
            "killer_insight": "When separate parts of an industry start solving adjacent bottlenecks at once, the story shifts from isolated progress to ecosystem readiness.",
            "contrarian_take": "The real signal is not hype volume but whether different constraints are easing at the same time.",
            "proof_points": [
                "Multiple sources point to simultaneous progress rather than one-off noise.",
                "The most important development is the change in system-level readiness.",
                "Readers should come away with a sharper frame, not just a recap.",
            ],
        }


def generate_json_from_prompt(prompt, prefill, max_tokens, schema_hint=None, retries=2):
    runtime = initialize_model_runtime()
    response = MLX_GENERATE(
        MODEL,
        TOKENIZER,
        prompt=prompt,
        max_tokens=max_tokens,
        draft_model=runtime["draft_model"],
        num_draft_tokens=runtime["num_draft_tokens"],
    )
    parsed = try_parse_model_json(response, prefill)
    if parsed is not None:
        return parsed

    last_response = response
    for _ in range(retries):
        repair_prompt = build_json_repair_prompt(last_response, prefill, schema_hint)
        repaired_response = generate_text_from_prompt(repair_prompt, max_tokens)
        parsed = try_parse_model_json(repaired_response, "")
        if parsed is not None:
            return parsed
        last_response = repaired_response

    raise ValueError("Model did not return valid JSON")


def generate_text_from_prompt(prompt, max_tokens):
    runtime = initialize_model_runtime()
    response = MLX_GENERATE(
        MODEL,
        TOKENIZER,
        prompt=prompt,
        max_tokens=max_tokens,
        draft_model=runtime["draft_model"],
        num_draft_tokens=runtime["num_draft_tokens"],
    )
    return response.strip()


def try_parse_model_json(response_text, prefill):
    candidates = []
    stripped = strip_code_fences(response_text.strip())
    if prefill:
        candidates.append(prefill + stripped)
    candidates.append(stripped)

    for candidate in candidates:
        parsed = try_parse_json_candidate(candidate)
        if parsed is not None:
            return parsed

    return None


def try_parse_json_candidate(text):
    clean_text_value = "".join(char for char in text if ord(char) >= 32)

    try:
        return json.loads(clean_text_value)
    except json.JSONDecodeError:
        pass

    extracted = extract_json_object(clean_text_value)
    if not extracted:
        return None

    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        return None


def extract_json_object(text):
    start = text.find("{")
    if start == -1:
        return ""

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return ""


def strip_code_fences(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def build_json_repair_prompt(response_text, prefill, schema_hint):
    expected_shape = schema_hint or "Return one valid JSON object."
    prompt = f"""<start_of_turn>user
Convert the following model output into one valid JSON object.

Expected JSON shape:
{expected_shape}

Original output:
\"\"\"
{prefill}{response_text.strip()}
\"\"\"

Rules:
- return valid JSON only
- do not include markdown fences
- do not include explanations
<end_of_turn>
<start_of_turn>model
"""
    return prompt


def save_run(plan, brief, depth, explanation_style, custom_style_instructions):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO newsletter_runs (
            brief,
            depth,
            explanation_style,
            custom_style_instructions,
            audience,
            tone,
            title,
            queries_json,
            sections_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            brief,
            depth,
            explanation_style,
            custom_style_instructions,
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
