"""
FAWP Scanner RSS/Atom feed generator.
Call generate_rss_feed(scan_results) to get an XML string.
Can be served from a FastAPI/Flask endpoint or written to a static file.
"""
from datetime import datetime, timezone


def generate_rss_feed(items: list, title="FAWP Scanner Alerts",
                      link="https://fawp-scanner.info",
                      description="Real-time FAWP regime detections") -> str:
    """
    items: list of dicts with keys:
      title, link, description, pub_date (ISO str), guid
    Returns RSS 2.0 XML string.
    """
    now = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
    items_xml = ""
    for item in items[:20]:
        pub = item.get("pub_date", now)
        items_xml += f"""
    <item>
      <title>{_esc(item.get('title','FAWP Alert'))}</title>
      <link>{item.get('link', link)}</link>
      <description>{_esc(item.get('description',''))}</description>
      <pubDate>{pub}</pubDate>
      <guid isPermaLink="false">{item.get('guid','')}</guid>
    </item>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>{_esc(title)}</title>
    <link>{link}</link>
    <description>{_esc(description)}</description>
    <lastBuildDate>{now}</lastBuildDate>
    <atom:link href="{link}/feed.xml" rel="self" type="application/rss+xml"/>
    {items_xml}
  </channel>
</rss>"""


def _esc(s: str) -> str:
    return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")


def scan_results_to_rss_items(wl_result=None, wx_result=None,
                               seis_result=None, dynamo_result=None) -> list:
    """Convert scanner session results to RSS items."""
    items = []
    now = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")

    if wl_result:
        for a in getattr(wl_result, "assets", []):
            if getattr(a, "regime_active", False):
                gap = float(getattr(a, "peak_gap_bits", 0) or 0)
                items.append({
                    "title":       f"🔴 FAWP: {a.ticker} [{a.timeframe}] — {gap:.4f} bits",
                    "link":        f"https://fawp-scanner.info/?scan=finance",
                    "description": (f"Finance scanner: {a.ticker} in FAWP regime. "
                                    f"Peak gap {gap:.4f} bits. τ⁺ₕ = "
                                    f"{getattr(getattr(a,'odw_result',None),'tau_h_plus','?')}."),
                    "pub_date":    now,
                    "guid":        f"fawp-finance-{a.ticker}-{a.timeframe}-{now[:10]}",
                })

    if wx_result and getattr(wx_result, "fawp_found", False):
        gap = float(getattr(wx_result, "peak_gap_bits", 0) or 0)
        loc = getattr(wx_result, "location", "?")
        items.append({
            "title":       f"🔴 FAWP Weather: {loc} — {gap:.4f} bits",
            "link":        "https://fawp-scanner.info/?scan=weather",
            "description": f"Weather scanner: {loc} in FAWP regime. Peak gap {gap:.4f} bits.",
            "pub_date":    now,
            "guid":        f"fawp-weather-{loc}-{now[:10]}",
        })

    if seis_result and getattr(seis_result, "fawp_found", False):
        gap = float(getattr(seis_result, "peak_gap_bits", 0) or 0)
        items.append({
            "title":       f"🔴 FAWP Seismic — {gap:.4f} bits",
            "link":        "https://fawp-scanner.info/?scan=seismic",
            "description": f"Seismic scanner in FAWP regime. Peak gap {gap:.4f} bits.",
            "pub_date":    now,
            "guid":        f"fawp-seismic-{now[:10]}",
        })

    return items
