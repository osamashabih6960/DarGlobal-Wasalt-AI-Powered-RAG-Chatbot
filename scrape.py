#!/usr/bin/env python3
"""Polite, robots-aware public-page crawler for DarGlobal and Wasalt.

Run locally where you have normal internet access:
  python -m pip install -r requirements.txt
  playwright install chromium
  python scrape.py --out data/full_dataset --max-pages 20000

The crawler does NOT log in, bypass CAPTCHAs, evade access controls, or call
undocumented/private APIs. It obeys robots.txt; if robots.txt cannot be read,
the host is skipped unless --allow-robots-unavailable is explicitly supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.robotparser
import xml.etree.ElementTree as ET
from collections import deque
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

SITES = {
    "DarGlobal": {
        "base": "https://darglobal.co.uk",
        "seeds": [
            "https://darglobal.co.uk/",
            "https://darglobal.co.uk/projects",
            "https://darglobal.co.uk/press",
            "https://darglobal.co.uk/investor-relations",
        ],
    },
    "Wasalt": {
        "base": "https://wasalt.sa",
        "seeds": [
            "https://wasalt.sa/en",
            "https://wasalt.sa/en/corporate",
            "https://wasalt.sa/en/international",
            "https://wasalt.sa/en/projects",
        ],
    },
}

UA = "DarWasaltResearchDatasetBot/1.0 (educational project; contact: add-your-email@example.com)"

ROBOTS = {}
LAST_REQ = {}

SKIP_EXT = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".pdf",
    ".zip",
    ".mp4",
    ".webm",
    ".woff",
    ".woff2",
    ".ttf",
    ".css",
    ".js",
)

SKIP_PARTS = (
    "/login",
    "/signup",
    "/register",
    "/account",
    "/profile",
    "/checkout",
    "/payment",
)


def norm_url(url):
    url, _ = urldefrag(url)
    p = urlparse(url)

    if p.scheme not in ("http", "https"):
        return None

    # Remove tracking fragments but preserve meaningful query parameters.
    return urlunparse(
        (
            p.scheme.lower(),
            p.netloc.lower(),
            p.path or "/",
            p.params,
            p.query,
            "",
        )
    )


def host_key(url):
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def robots_for(url, session, allow_unavailable=False):
    host = host_key(url)

    if host not in ROBOTS:
        rp = urllib.robotparser.RobotFileParser()
        robots_url = host + "/robots.txt"

        try:
            response = session.get(robots_url, timeout=20)

            if response.status_code == 200:
                rp.parse(response.text.splitlines())
                ROBOTS[host] = (rp, True, response.text)
            else:
                ROBOTS[host] = (rp, False, "")

        except requests.RequestException:
            ROBOTS[host] = (rp, False, "")

    rp, ok, raw = ROBOTS[host]

    if not ok and not allow_unavailable:
        return False, []

    try:
        allowed = rp.can_fetch(UA, url) if ok else True
    except Exception:
        allowed = False

    sitemaps = []

    for line in raw.splitlines():
        if line.lower().startswith("sitemap:"):
            sitemaps.append(line.split(":", 1)[1].strip())

    return allowed, sitemaps


def polite_get(url, session, delay, allow_unavailable=False):
    ok, _ = robots_for(url, session, allow_unavailable)

    if not ok:
        return None

    host = host_key(url)
    wait = max(0, delay - (time.time() - LAST_REQ.get(host, 0)))

    if wait:
        time.sleep(wait)

    try:
        response = session.get(
            url,
            timeout=30,
            allow_redirects=True,
        )

        LAST_REQ[host] = time.time()

        if response.status_code in (401, 403, 404, 410, 429):
            return None

        if response.status_code >= 500:
            return None

        response.raise_for_status()
        return response

    except requests.RequestException:
        LAST_REQ[host] = time.time()
        return None


def parse_sitemap(
    url,
    session,
    delay,
    allow_unavailable,
    seen=None,
):
    if seen is None:
        seen = set()

    if url in seen:
        return []

    seen.add(url)

    response = polite_get(
        url,
        session,
        delay,
        allow_unavailable,
    )

    if not response:
        return []

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
        return []

    tag = root.tag.rsplit("}", 1)[-1].lower()

    locs = [
        (element.text or "").strip()
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1].lower() == "loc" and element.text
    ]

    if tag == "sitemapindex":
        output = []

        for loc in locs:
            output.extend(
                parse_sitemap(
                    loc,
                    session,
                    delay,
                    allow_unavailable,
                    seen,
                )
            )

        return output

    return locs


def extract_jsonld(soup):
    output = []

    for tag in soup.find_all(
        "script",
        type="application/ld+json",
    ):
        raw = tag.string or tag.get_text()

        try:
            obj = json.loads(raw)
            stack = [obj]

            while stack:
                item = stack.pop()

                if isinstance(item, dict):
                    if item.get("@type") or item.get("name") or item.get("offers"):
                        output.append(item)

                    stack.extend(
                        value for value in item.values() if isinstance(value, (dict, list))
                    )

                elif isinstance(item, list):
                    stack.extend(item)

        except Exception:
            pass

    return output


def flatten_structured(obj, limit=60):
    parts = []

    def walk(item, prefix=""):
        if len(parts) >= limit:
            return

        if isinstance(item, dict):
            for key, value in item.items():
                if key in (
                    "image",
                    "thumbnailUrl",
                    "logo",
                    "@context",
                ):
                    continue

                walk(value, f"{prefix}{key}.")

        elif isinstance(item, list):
            for value in item[:20]:
                walk(value, prefix)

        elif isinstance(item, (str, int, float)) and str(item).strip():
            value = re.sub(r"\s+", " ", str(item)).strip()

            if len(value) > 2:
                parts.append(f"{prefix[:-1]}: {value[:1000]}")

    walk(obj)

    return "; ".join(parts)


def extract_page(html, url, company):
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    desc = soup.find(
        "meta",
        attrs={"name": "description"},
    )

    meta = desc.get("content", "").strip() if desc else ""

    canonical = soup.find(
        "link",
        rel="canonical",
    )

    canonical_url = (
        norm_url(urljoin(url, canonical.get("href")))
        if canonical and canonical.get("href")
        else url
    )

    # Remove boilerplate and non-content elements.
    for element in soup(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "iframe",
            "form",
            "header",
            "footer",
            "nav",
        ]
    ):
        element.decompose()

    headings = []

    for heading in soup.find_all(["h1", "h2", "h3"]):
        heading_text = re.sub(
            r"\s+",
            " ",
            heading.get_text(" ", strip=True),
        )

        if heading_text and heading_text not in headings:
            headings.append(heading_text)

    chunks = []
    current = title or "Page"
    buffer = []

    for element in soup.find_all(["h1", "h2", "h3", "p", "li", "td", "dt", "dd"]):
        text = re.sub(
            r"\s+",
            " ",
            element.get_text(" ", strip=True),
        )

        if len(text) < 4:
            continue

        if element.name in ("h1", "h2", "h3"):
            if buffer:
                chunks.append((current, " ".join(buffer)))
                buffer = []

            current = text

        elif len(text) > 15:
            buffer.append(text)

    if buffer:
        chunks.append((current, " ".join(buffer)))

    # Collect structured data as separately attributed records.
    structured = extract_jsonld(soup)

    links = []

    for anchor in soup.find_all("a", href=True):
        link_url = norm_url(urljoin(canonical_url, anchor["href"]))

        if not link_url:
            continue

        parsed = urlparse(link_url)
        canonical_host = urlparse(canonical_url).netloc.lower()

        if parsed.netloc.lower() != canonical_host:
            continue

        if parsed.path.lower().endswith(SKIP_EXT):
            continue

        if any(part in parsed.path.lower() for part in SKIP_PARTS):
            continue

        links.append(link_url)

    rows = []

    category = (
        "listing"
        if re.search(
            r"(property|properties|listing|for-sale|for-rent|rent|buy)",
            canonical_url,
            re.I,
        )
        else "project"
        if "project" in canonical_url.lower()
        else "news"
        if re.search(
            r"(press|news|media|announcement)",
            canonical_url,
            re.I,
        )
        else "page"
    )

    if meta:
        rows.append(
            {
                "company": company,
                "category": category,
                "title": (title + " - meta")[:240],
                "text": meta,
                "source_url": canonical_url,
            }
        )

    for heading, body in chunks:
        if len(body) > 35:
            rows.append(
                {
                    "company": company,
                    "category": category,
                    "title": (title + " - " + heading)[:240],
                    "text": body[:12000],
                    "source_url": canonical_url,
                }
            )

    for obj in structured:
        body = flatten_structured(obj)

        if len(body) > 60:
            name = obj.get("name") or obj.get("headline") or title

            item_type = obj.get("@type", "structured_data")

            rows.append(
                {
                    "company": company,
                    "category": ("structured_" + str(item_type).lower()),
                    "title": str(name)[:240],
                    "text": body[:12000],
                    "source_url": canonical_url,
                }
            )

    return rows, links, canonical_url


def browser_fetch(url, delay, allow_unavailable=False):
    """Optional JS-rendered fallback; obeys robots and host delay."""

    ok, _ = robots_for(
        url,
        SESS,
        allow_unavailable,
    )

    if not ok:
        return None

    host = host_key(url)

    wait = max(
        0,
        delay - (time.time() - LAST_REQ.get(host, 0)),
    )

    if wait:
        time.sleep(wait)

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
            )

            page = browser.new_page(
                user_agent=UA,
            )

            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=45000,
            )

            page.wait_for_timeout(2500)

            html = page.content()
            final_url = page.url

            browser.close()

        LAST_REQ[host] = time.time()

        return type(
            "RenderedResponse",
            (),
            {
                "text": html,
                "url": final_url,
            },
        )()

    except Exception as exc:
        print(
            f"[browser fallback failed] {url}: {exc}",
            file=sys.stderr,
        )

        LAST_REQ[host] = time.time()
        return None


SESS = None


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--out",
        default="data/full_dataset",
    )

    ap.add_argument(
        "--only",
        choices=["DarGlobal", "Wasalt"],
    )

    ap.add_argument(
        "--max-pages",
        type=int,
        default=20000,
    )

    ap.add_argument(
        "--delay",
        type=float,
        default=1.5,
    )

    ap.add_argument(
        "--allow-robots-unavailable",
        action="store_true",
        help=(
            "Only use if you independently verified the site's "
            "crawl policy; default is fail closed."
        ),
    )

    ap.add_argument(
        "--no-playwright",
        action="store_true",
    )

    args = ap.parse_args()

    os.makedirs(
        os.path.dirname(args.out) or ".",
        exist_ok=True,
    )

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": UA,
            "Accept-Language": "en-US,en;q=0.9",
        }
    )

    global SESS
    SESS = session

    companies = [args.only] if args.only else list(SITES)

    all_rows = []
    crawled = set()
    discovered = set()
    blocked = 0

    for company in companies:
        config = SITES[company]
        base = config["base"]
        host = urlparse(base).netloc
        seeds = config["seeds"][:]

        # Sitemap URLs are discovered only from robots.txt.
        _, sitemap_urls = robots_for(
            base + "/",
            session,
            args.allow_robots_unavailable,
        )

        urls = []

        for sitemap in sitemap_urls:
            urls.extend(
                parse_sitemap(
                    sitemap,
                    session,
                    args.delay,
                    args.allow_robots_unavailable,
                )
            )

        if not urls and args.allow_robots_unavailable:
            urls.extend(
                parse_sitemap(
                    base + "/sitemap.xml",
                    session,
                    args.delay,
                    True,
                )
            )

        queue = deque()

        for url in urls + seeds:
            normalized = norm_url(url)

            if normalized and urlparse(normalized).netloc == host and normalized not in discovered:
                discovered.add(normalized)
                queue.append((normalized, 0))

        print(
            f"[{company}] sitemap urls discovered: {len(urls)}; starting queue: {len(queue)}",
            flush=True,
        )

        while queue and len(crawled) < args.max_pages:
            url, depth = queue.popleft()

            if url in crawled:
                continue

            allowed, _ = robots_for(
                url,
                session,
                args.allow_robots_unavailable,
            )

            if not allowed:
                blocked += 1
                continue

            response = polite_get(
                url,
                session,
                args.delay,
                args.allow_robots_unavailable,
            )

            if not response and not args.no_playwright:
                response = browser_fetch(
                    url,
                    args.delay,
                    args.allow_robots_unavailable,
                )

            elif response and not args.no_playwright:
                # Render SPA shells that contain very little text.
                quick_text = BeautifulSoup(
                    response.text,
                    "lxml",
                ).get_text(" ", strip=True)

                if len(quick_text) < 250:
                    rendered = browser_fetch(
                        url,
                        args.delay,
                        args.allow_robots_unavailable,
                    )

                    if rendered:
                        response = rendered

            if not response:
                continue

            final_url = norm_url(response.url)

            if not final_url or urlparse(final_url).netloc != host:
                continue

            crawled.add(url)

            rows, links, canonical = extract_page(
                response.text,
                final_url,
                company,
            )

            all_rows.extend(rows)

            # Discover same-host links while limiting query explosion.
            for link in links:
                parsed = urlparse(link)

                if parsed.netloc != host or link in discovered:
                    continue

                if len(parsed.query) > 300:
                    continue

                discovered.add(link)
                queue.append((link, depth + 1))

            if len(crawled) % 50 == 0:
                print(
                    f"[{company}] crawled={len(crawled)} "
                    f"discovered={len(discovered)} "
                    f"records={len(all_rows)}",
                    flush=True,
                )

        print(
            f"[{company}] finished; total crawled so far={len(crawled)}",
            flush=True,
        )

    # Deduplicate by source URL and content hash.
    # Keep distinct sections from the same page.
    output = []
    seen = set()

    for row in all_rows:
        body = re.sub(
            r"\s+",
            " ",
            row["text"],
        ).strip()

        key = (
            row["company"],
            row["source_url"],
            hashlib.sha256(body.encode()).hexdigest(),
        )

        if len(body) < 25 or key in seen:
            continue

        seen.add(key)

        output.append(
            {
                **row,
                "text": body,
            }
        )

    columns = [
        "id",
        "company",
        "category",
        "title",
        "text",
        "source_url",
    ]

    for index, row in enumerate(output, 1):
        row["id"] = f"dw-{index:07d}"

    root = args.out

    for extension in ("json", "jsonl", "csv"):
        path = f"{root}.{extension}"

        if extension == "json":
            with open(
                path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    output,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

        elif extension == "jsonl":
            with open(
                path,
                "w",
                encoding="utf-8",
            ) as file:
                for row in output:
                    file.write(
                        json.dumps(
                            {key: row[key] for key in columns},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        else:
            with open(
                path,
                "w",
                encoding="utf-8-sig",
                newline="",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=columns,
                )

                writer.writeheader()

                writer.writerows({key: row[key] for key in columns} for row in output)

    manifest = {
        "records": len(output),
        "pages_crawled": len(crawled),
        "urls_discovered": len(discovered),
        "robots_blocked": blocked,
        "companies": companies,
        "generated_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "user_agent": UA,
        "max_pages": args.max_pages,
        "note": (
            "Only public pages accessible under robots.txt were "
            "included. This is a crawl snapshot, not a guarantee "
            "of every database listing."
        ),
    }

    with open(
        root + "_manifest.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print(
        json.dumps(
            manifest,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
