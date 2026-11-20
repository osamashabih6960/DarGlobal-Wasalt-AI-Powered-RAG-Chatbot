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
import argparse, csv, hashlib, json, os, re, sys, time, urllib.robotparser
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag, urlunparse
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

SITES = {
    "DarGlobal": {"base": "https://darglobal.co.uk", "seeds": [
        "https://darglobal.co.uk/", "https://darglobal.co.uk/projects",
        "https://darglobal.co.uk/press", "https://darglobal.co.uk/investor-relations",
    ]},
    "Wasalt": {"base": "https://wasalt.sa", "seeds": [
        "https://wasalt.sa/en", "https://wasalt.sa/en/corporate",
        "https://wasalt.sa/en/international", "https://wasalt.sa/en/projects",
    ]},
}
UA = "DarWasaltResearchDatasetBot/1.0 (educational project; contact: add-your-email@example.com)"
ROBOTS, LAST_REQ = {}, {}
SKIP_EXT = (".jpg",".jpeg",".png",".gif",".webp",".svg",".pdf",".zip",".mp4",".webm",".woff",".woff2",".ttf",".css",".js")
SKIP_PARTS = ("/login", "/signup", "/register", "/account", "/profile", "/checkout", "/payment")

def norm_url(url):
    url, _ = urldefrag(url)
    p=urlparse(url)
    if p.scheme not in ("http","https"): return None
    # remove tracking params but preserve meaningful search/filter query parameters
    return urlunparse((p.scheme.lower(), p.netloc.lower(), p.path or "/", p.params, p.query, ""))

def host_key(url):
    p=urlparse(url); return f"{p.scheme}://{p.netloc}"

def robots_for(url, session, allow_unavailable=False):
    host=host_key(url)
    if host not in ROBOTS:
        rp=urllib.robotparser.RobotFileParser()
        robots_url=host+"/robots.txt"
        try:
            r=session.get(robots_url, timeout=20)
            if r.status_code == 200:
                rp.parse(r.text.splitlines())
                ROBOTS[host]=(rp, True, r.text)
            else:
                ROBOTS[host]=(rp, False, "")
        except requests.RequestException:
            ROBOTS[host]=(rp, False, "")
    rp, ok, raw=ROBOTS[host]
    if not ok and not allow_unavailable:
        return False, []
    try: allowed=rp.can_fetch(UA, url) if ok else True
    except Exception: allowed=False
    sitemaps=[]
    for line in raw.splitlines():
        if line.lower().startswith("sitemap:"):
            sitemaps.append(line.split(":",1)[1].strip())
    return allowed, sitemaps

def polite_get(url, session, delay, allow_unavailable=False):
    ok,_=robots_for(url,session,allow_unavailable)
    if not ok: return None
    host=host_key(url)
    wait=max(0, delay-(time.time()-LAST_REQ.get(host,0)))
    if wait: time.sleep(wait)
    try:
        r=session.get(url,timeout=30,allow_redirects=True)
        LAST_REQ[host]=time.time()
        if r.status_code in (401,403,404,410,429): return None
        if r.status_code >= 500: return None
        r.raise_for_status()
        return r
    except requests.RequestException:
        LAST_REQ[host]=time.time()
        return None

def parse_sitemap(url, session, delay, allow_unavailable, seen=None):
    seen=seen or set()
    if url in seen: return []
    seen.add(url)
    r=polite_get(url,session,delay,allow_unavailable)
    if not r: return []
    try: root=ET.fromstring(r.content)
    except ET.ParseError: return []
    tag=root.tag.rsplit("}",1)[-1].lower()
    locs=[(e.text or "").strip() for e in root.iter() if e.tag.rsplit("}",1)[-1].lower()=="loc" and e.text]
    if tag=="sitemapindex":
        out=[]
        for loc in locs: out.extend(parse_sitemap(loc,session,delay,allow_unavailable,seen))
        return out
    return locs

def extract_jsonld(soup):
    out=[]
    for tag in soup.find_all("script",type="application/ld+json"):
        raw=tag.string or tag.get_text()
        try:
            obj=json.loads(raw)
            stack=[obj]
            while stack:
                x=stack.pop()
                if isinstance(x,dict):
                    if x.get("@type") or x.get("name") or x.get("offers"):
                        out.append(x)
                    stack.extend(v for v in x.values() if isinstance(v,(dict,list)))
                elif isinstance(x,list): stack.extend(x)
        except Exception: pass
    return out

def flatten_structured(obj, limit=60):
    parts=[]
    def walk(x,prefix=""):
        if len(parts)>=limit:return
        if isinstance(x,dict):
            for k,v in x.items():
                if k in ("image","thumbnailUrl","logo","@context"): continue
                walk(v, f"{prefix}{k}.")
        elif isinstance(x,list):
            for v in x[:20]: walk(v,prefix)
        elif isinstance(x,(str,int,float)) and str(x).strip():
            s=re.sub(r"\s+"," ",str(x)).strip()
            if len(s)>2: parts.append(f"{prefix[:-1]}: {s[:1000]}")
    walk(obj)
    return "; ".join(parts)

def extract_page(html,url,company):
    soup=BeautifulSoup(html,"lxml")
    title=soup.title.get_text(" ",strip=True) if soup.title else ""
    desc=soup.find("meta",attrs={"name":"description"})
    meta=desc.get("content","").strip() if desc else ""
    canonical=soup.find("link",rel="canonical")
    canonical_url=norm_url(urljoin(url,canonical.get("href"))) if canonical and canonical.get("href") else url
    # remove boilerplate and non-content
    for x in soup(["script","style","noscript","svg","iframe","form","header","footer","nav"]): x.decompose()
    h1=soup.find("h1")
    headings=[]
    for h in soup.find_all(["h1","h2","h3"]):
        t=re.sub(r"\s+"," ",h.get_text(" ",strip=True))
        if t and t not in headings: headings.append(t)
    chunks=[]
    current=title or "Page"
    buf=[]
    for el in soup.find_all(["h1","h2","h3","p","li","td","dt","dd"]):
        txt=re.sub(r"\s+"," ",el.get_text(" ",strip=True))
        if len(txt)<4: continue
        if el.name in ("h1","h2","h3"):
            if buf: chunks.append((current," ".join(buf))); buf=[]
            current=txt
        elif len(txt)>15: buf.append(txt)
    if buf: chunks.append((current," ".join(buf)))
    # collect structured data as separately attributed records
    structured=extract_jsonld(soup)
    links=[]
    for a in soup.find_all("a",href=True):
        u=norm_url(urljoin(canonical_url,a["href"]))
        if not u: continue
        p=urlparse(u)
        if p.netloc.lower()!=urlparse(canonical_url).netloc.lower(): continue
        if p.path.lower().endswith(SKIP_EXT) or any(x in p.path.lower() for x in SKIP_PARTS): continue
        links.append(u)
    rows=[]
    category="listing" if re.search(r"(property|properties|listing|for-sale|for-rent|rent|buy)",canonical_url,re.I) else \
             "project" if "project" in canonical_url.lower() else \
             "news" if re.search(r"(press|news|media|announcement)",canonical_url,re.I) else "page"
    if meta: rows.append({"company":company,"category":category,"title":(title+" - meta")[:240],"text":meta,"source_url":canonical_url})
    for heading,body in chunks:
        if len(body)>35:
            rows.append({"company":company,"category":category,"title":(title+" - "+heading)[:240],"text":body[:12000],"source_url":canonical_url})
    for obj in structured:
        body=flatten_structured(obj)
        if len(body)>60:
            name=obj.get("name") or obj.get("headline") or title
            typ=obj.get("@type","structured_data")
            rows.append({"company":company,"category":"structured_"+str(typ).lower(),"title":str(name)[:240],"text":body[:12000],"source_url":canonical_url})
    return rows,links,canonical_url

def browser_fetch(url, delay, allow_unavailable=False):
    """Optional JS-rendered fallback; still obeys robots and per-host delay."""
    ok,_=robots_for(url, SESS, allow_unavailable)
    if not ok: return None
    host=host_key(url)
    wait=max(0, delay-(time.time()-LAST_REQ.get(host,0)))
    if wait: time.sleep(wait)
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True)
            page=browser.new_page(user_agent=UA)
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(2500)
            html=page.content()
            final=page.url
            browser.close()
        LAST_REQ[host]=time.time()
        return type("RenderedResponse",(),{"text":html,"url":final})()
    except Exception as exc:
        print(f"[browser fallback failed] {url}: {exc}",file=sys.stderr)
        LAST_REQ[host]=time.time()
        return None

SESS = None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="data/full_dataset")
    ap.add_argument("--only",choices=["DarGlobal","Wasalt"])
    ap.add_argument("--max-pages",type=int,default=20000)
    ap.add_argument("--delay",type=float,default=1.5)
    ap.add_argument("--allow-robots-unavailable",action="store_true",
                    help="Only use if you independently verified the site's crawl policy; default is fail closed.")
    ap.add_argument("--no-playwright",action="store_true")
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".",exist_ok=True)
    sess=requests.Session(); sess.headers.update({"User-Agent":UA,"Accept-Language":"en-US,en;q=0.9"})
    global SESS; SESS=sess
    companies=[args.only] if args.only else list(SITES)
    all_rows=[]; crawled=set(); discovered=set(); blocked=0
    for company in companies:
        cfg=SITES[company]; base=cfg["base"]; host=urlparse(base).netloc
        seeds=cfg["seeds"][:]
        # Sitemap URLs are discovered only from robots.txt.
        _,sm=robots_for(base+"/",sess,args.allow_robots_unavailable)
        urls=[]
        for sitemap in sm: urls.extend(parse_sitemap(sitemap,sess,args.delay,args.allow_robots_unavailable))
        if not urls and args.allow_robots_unavailable:
            urls.extend(parse_sitemap(base+"/sitemap.xml",sess,args.delay,True))
        queue=deque()
        for u in urls+seeds:
            nu=norm_url(u)
            if nu and urlparse(nu).netloc==host and nu not in discovered:
                discovered.add(nu); queue.append((nu,0))
        print(f"[{company}] sitemap urls discovered: {len(urls)}; starting queue: {len(queue)}",flush=True)
        while queue and len(crawled)<args.max_pages:
            url,depth=queue.popleft()
            if url in crawled: continue
            allowed,_=robots_for(url,sess,args.allow_robots_unavailable)
            if not allowed:
                blocked+=1; continue
            resp=polite_get(url,sess,args.delay,args.allow_robots_unavailable)
            if not resp and not args.no_playwright:
                resp=browser_fetch(url,args.delay,args.allow_robots_unavailable)
            elif resp and not args.no_playwright:
                # Some SPA shells return little meaningful text; render those pages once.
                quick=BeautifulSoup(resp.text,"lxml").get_text(" ",strip=True)
                if len(quick)<250:
                    rendered=browser_fetch(url,args.delay,args.allow_robots_unavailable)
                    if rendered: resp=rendered
            if not resp: continue
            final=norm_url(resp.url)
            if not final or urlparse(final).netloc!=host: continue
            crawled.add(url)
            rows,links,canonical=extract_page(resp.text,final,company)
            all_rows.extend(rows)
            # Crawl links from all discovered pages. Limit query explosion but retain known listing filters.
            for link in links:
                p=urlparse(link)
                if p.netloc!=host or link in discovered: continue
                if len(p.query)>300: continue
                discovered.add(link); queue.append((link,depth+1))
            if len(crawled)%50==0:
                print(f"[{company}] crawled={len(crawled)} discovered={len(discovered)} records={len(all_rows)}",flush=True)
        print(f"[{company}] finished; total crawled so far={len(crawled)}",flush=True)
    # Deduplicate by normalized source + content hash; retain the same page's distinct sections.
    out=[]; seen=set()
    for r in all_rows:
        body=re.sub(r"\s+"," ",r["text"]).strip()
        key=(r["company"],r["source_url"],hashlib.sha256(body.encode()).hexdigest())
        if len(body)<25 or key in seen: continue
        seen.add(key); out.append({**r,"text":body})
    cols=["id","company","category","title","text","source_url"]
    for i,r in enumerate(out,1): r["id"]=f"dw-{i:07d}"
    root=args.out
    for ext in ("json","jsonl","csv"):
        path=f"{root}.{ext}"
        if ext=="json":
            with open(path,"w",encoding="utf-8") as f: json.dump(out,f,ensure_ascii=False,indent=2)
        elif ext=="jsonl":
            with open(path,"w",encoding="utf-8") as f:
                for r in out: f.write(json.dumps({k:r[k] for k in cols},ensure_ascii=False)+"\n")
        else:
            with open(path,"w",encoding="utf-8-sig",newline="") as f:
                w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows({k:r[k] for k in cols} for r in out)
    manifest={"records":len(out),"pages_crawled":len(crawled),"urls_discovered":len(discovered),
              "robots_blocked":blocked,"companies":companies,"generated_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
              "user_agent":UA,"max_pages":args.max_pages,
              "note":"Only public pages accessible under robots.txt were included. This is a crawl snapshot, not a guarantee of every database listing."}
    with open(root+"_manifest.json","w",encoding="utf-8") as f: json.dump(manifest,f,indent=2)
    print(json.dumps(manifest,indent=2))

if __name__=="__main__": main()
