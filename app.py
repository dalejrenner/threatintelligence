import re
import time
import html
import hashlib
import feedparser
import requests
from bs4 import BeautifulSoup

import streamlit as st
from transformers import pipeline


# ----------------------------- Page & Constants -----------------------------
st.set_page_config(page_title="Threat Intel Dashboard", layout="wide")
st.title("🛡️ Threat Intel Dashboard")
st.caption("Pulls recent security headlines, enriches with ML, and buckets by industry.")

DEFAULT_FEEDS = [
    # News & research
    "https://www.bleepingcomputer.com/feed/",
    "https://www.darkreading.com/rss.xml",
    "https://krebsonsecurity.com/feed/",
    "https://www.securityweek.com/feed/",
    "https://therecord.media/feed/",
    # Advisories
    "https://www.cisa.gov/cybersecurity-advisories/all.xml",
    "https://msrc-blog.microsoft.com/feed/",
    "https://www.us-cert.gov/ncas/alerts.xml",
]

INDUSTRY_LABELS = ["Financial", "Technology", "Retail"]
THREAT_LABELS = ["Ransomware", "Phishing", "Vulnerability", "Data Breach", "Malware", "Supply Chain"]

# Regex helpers (very lightweight IOC & CVE pulls)
CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
SHA256_RE = re.compile(r"\b[a-fA-F0-9]{64}\b")
DOMAIN_RE = re.compile(r"\b([a-z0-9-]+\.)+[a-z]{2,}\b", re.I)


# ----------------------------- Caching: Models ------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    # Zero-shot for flexible industry & threat classification
    clf_industry = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    clf_threat = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return clf_industry, clf_threat

clf_industry, clf_threat = load_models()


# --------------------------- Utility: Fetch & Clean -------------------------
@st.cache_data(show_spinner=False, ttl=900)  # cache feeds for 15 minutes
def fetch_items(feed_urls):
    items = []
    for url in feed_urls:
        try:
            parsed = feedparser.parse(url)
            for e in parsed.entries[:30]:
                title = html.unescape(e.get("title", "")).strip()
                link = e.get("link", "")
                summary_html = e.get("summary", e.get("description", "")) or ""
                summary = BeautifulSoup(summary_html, "html.parser").get_text(" ", strip=True)

                # Try to pull a bit of article text if possible (best-effort; keep fast)
                text = summary
                try:
                    if link and link.startswith("http"):
                        resp = requests.get(link, timeout=6)
                        if resp.ok and "text/html" in resp.headers.get("Content-Type", ""):
                            soup = BeautifulSoup(resp.text, "html.parser")
                            # crude extract: pick largest paragraph blob
                            paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
                            paras = [p for p in paras if len(p) > 60]
                            if paras:
                                text = " ".join(paras[:6])  # keep it short for speed
                except Exception:
                    pass

                published = (
                    e.get("published", "")
                    or e.get("updated", "")
                    or ""
                )

                # simple stable id for de-dup
                key = hashlib.sha1((title + link).encode("utf-8")).hexdigest()

                items.append(
                    {
                        "id": key,
                        "source": parsed.feed.get("title", url),
                        "title": title,
                        "link": link,
                        "published": published,
                        "summary": summary[:500],
                        "text": text[:4000],
                    }
                )
        except Exception as ex:
            st.warning(f"Failed to parse {url}: {ex}")
    # de-dup by id
    seen = set()
    unique = []
    for it in items:
        if it["id"] in seen:
            continue
        seen.add(it["id"])
        unique.append(it)
    return unique


def extract_iocs(text):
    cves = sorted(set(CVE_RE.findall(text)))
    ips = sorted(set(IP_RE.findall(text)))
    shas = sorted(set(SHA256_RE.findall(text)))
    domains = sorted(set(DOMAIN_RE.findall(text)))
    return cves, ips, shas, domains


# ---------------------------- ML Enrichment Step ----------------------------
def classify_industry(text: str):
    res = clf_industry(text, INDUSTRY_LABELS, multi_label=False)
    top_label = res["labels"][0]
    top_score = float(res["scores"][0])
    return top_label, top_score, dict(zip(res["labels"], [float(s) for s in res["scores"]]))


def classify_threat(text: str):
    res = clf_threat(text, THREAT_LABELS, multi_label=True)  # allow multi-label for threat types
    # pick best + full distribution
    top_label = res["labels"][0]
    top_score = float(res["scores"][0])
    dist = dict(zip(res["labels"], [float(s) for s in res["scores"]]))
    return top_label, top_score, dist


@st.cache_data(show_spinner=True, ttl=900)
def enrich(items):
    enriched = []
    for it in items:
        blob = (it["title"] + ". " + it["text"]).strip()
        try:
            ind_label, ind_score, ind_dist = classify_industry(blob)
            thr_label, thr_score, thr_dist = classify_threat(blob)
        except Exception as ex:
            ind_label, ind_score, ind_dist = "Unknown", 0.0, {}
            thr_label, thr_score, thr_dist = "Unknown", 0.0, {}

        cves, ips, shas, domains = extract_iocs(blob)
        enriched.append(
            {
                **it,
                "industry": ind_label,
                "industry_conf": round(ind_score, 3),
                "industry_dist": ind_dist,
                "threat": thr_label,
                "threat_conf": round(thr_score, 3),
                "threat_dist": thr_dist,
                "cves": cves,
                "ips": ips,
                "hashes": shas,
                "domains": [d for d in domains if not d.endswith((".jpg", ".png", ".gif"))],
            }
        )
    return enriched


# --------------------------------- Sidebar ----------------------------------
with st.sidebar:
    st.subheader("Settings")
    feeds = st.text_area(
        "RSS feeds (one per line)",
        value="\n".join(DEFAULT_FEEDS),
        height=200,
        help="Add or remove feeds. Press Refresh after changes."
    ).splitlines()

    colA, colB = st.columns(2)
    with colA:
        refresh = st.button("🔄 Refresh")
    with colB:
        show_iocs = st.toggle("Show IOCs", value=True)

    st.markdown("---")
    industry_filter = st.multiselect("Industry Filter", INDUSTRY_LABELS, default=INDUSTRY_LABELS)
    threat_filter = st.multiselect("Threat Filter", THREAT_LABELS, default=THREAT_LABELS)
    min_conf = st.slider("Min Industry Confidence", 0.0, 1.0, 0.5, 0.05)


# ------------------------------ Main Execution ------------------------------
if refresh or "cache_busted" not in st.session_state:
    st.session_state["cache_busted"] = time.time()
    items = fetch_items(tuple(feeds))
    data = enrich(items)
else:
    items = fetch_items(tuple(feeds))
    data = enrich(items)

# Filter view
view = [
    d for d in data
    if d["industry"] in industry_filter
    and d["threat"] in threat_filter
    and d["industry_conf"] >= min_conf
]

# ------------------------------ Overview Counts -----------------------------
st.subheader("Overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Items", len(data))
m2.metric("Filtered View", len(view))
m3.metric("Unique CVEs", len({c for d in data for c in d["cves"]}))
m4.metric("Sources", len(set(d["source"] for d in data)))

# ------------------------------ Tabs by Industry ----------------------------
tabs = st.tabs([f"{ind}" for ind in INDUSTRY_LABELS])

for t_ind, tab in zip(INDUSTRY_LABELS, tabs):
    with tab:
        subset = [d for d in view if d["industry"] == t_ind]
        st.write(f"**{len(subset)} items** for **{t_ind}**")
        for d in subset:
            with st.container(border=True):
                st.markdown(f"### [{d['title']}]({d['link']})")
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(d["summary"] or d["text"][:300] + "...")
                    st.caption(f"Source: {d['source']}  •  Published: {d['published']}")
                with col2:
                    st.markdown(f"**Threat:** {d['threat']} ({d['threat_conf']:.2f})")
                    st.markdown(f"**Industry conf:** {d['industry_conf']:.2f}")

                    # Small distribution printouts
                    ind_dist_sorted = sorted(d["industry_dist"].items(), key=lambda x: -x[1])[:3]
                    thr_dist_sorted = sorted(d["threat_dist"].items(), key=lambda x: -x[1])[:3]
                    st.caption("Industry scores:")
                    for k, v in ind_dist_sorted:
                        st.text(f"{k}: {v:.2f}")
                    st.caption("Threat scores:")
                    for k, v in thr_dist_sorted:
                        st.text(f"{k}: {v:.2f}")

                if show_iocs:
                    if any([d["cves"], d["ips"], d["hashes"], d["domains"]]):
                        with st.expander("Indicators & Entities"):
                            if d["cves"]:
                                st.write("**CVEs:** ", ", ".join(d["cves"]))
                            if d["ips"]:
                                st.write("**IPs:** ", ", ".join(d["ips"][:10]))
                            if d["hashes"]:
                                st.write("**SHA256:** ", ", ".join(d["hashes"][:5]))
                            if d["domains"]:
                                st.write("**Domains:** ", ", ".join(d["domains"][:10]))