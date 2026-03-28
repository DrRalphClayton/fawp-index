"""
Email digest module for FAWP Scanner.
Sends a daily summary of all active FAWP signals via Resend API (free tier).
"""
import urllib.request, json

RESEND_API_URL = "https://api.resend.com/emails"


def _build_html(finance_results=None, weather_result=None,
                seismic_result=None, dynamo_result=None) -> str:
    rows = []
    if finance_results:
        for a in getattr(finance_results, "assets", []):
            if getattr(a, "regime_active", False):
                rows.append(f"<tr><td>📈 Finance</td><td>{a.ticker}</td>"
                            f"<td style='color:#C0111A'>FAWP</td>"
                            f"<td>{a.peak_gap_bits:.4f}</td></tr>")
    if weather_result and getattr(weather_result, "fawp_found", False):
        rows.append(f"<tr><td>🌦 Weather</td>"
                    f"<td>{getattr(weather_result,'location','?')}</td>"
                    f"<td style='color:#C0111A'>FAWP</td>"
                    f"<td>{getattr(weather_result,'peak_gap_bits',0):.4f}</td></tr>")
    if seismic_result and getattr(seismic_result, "fawp_found", False):
        rows.append(f"<tr><td>🌍 Seismic</td><td>Region</td>"
                    f"<td style='color:#C0111A'>FAWP</td>"
                    f"<td>{getattr(seismic_result,'peak_gap_bits',0):.4f}</td></tr>")
    if dynamo_result:
        _dodw = dynamo_result.get("odw")
        if _dodw and getattr(_dodw, "fawp_found", False):
            rows.append(f"<tr><td>⚙️ Dynamic</td>"
                        f"<td>{dynamo_result.get('domain','Custom')}</td>"
                        f"<td style='color:#C0111A'>FAWP</td>"
                        f"<td>{getattr(_dodw,'peak_gap_bits',0):.4f}</td></tr>")

    if not rows:
        body = "<p>✅ No active FAWP signals across all scanners.</p>"
    else:
        body = (
            "<table border='1' cellpadding='6' style='border-collapse:collapse'>"
            "<tr><th>Scanner</th><th>Asset</th><th>Status</th><th>Peak Gap (bits)</th></tr>"
            + "".join(rows) + "</table>"
        )

    return f"""<html><body>
<h2>🔴 FAWP Scanner Daily Digest</h2>
{body}
<hr><p style='color:grey;font-size:.8em'>
fawp-index · <a href='https://fawp-scanner.info'>fawp-scanner.info</a>
</p></body></html>"""


def send_digest(to_email: str, api_key: str, from_email: str = "digest@fawp-scanner.info",
                finance_results=None, weather_result=None,
                seismic_result=None, dynamo_result=None) -> bool:
    """Send a FAWP digest email via Resend. Returns True on success."""
    html = _build_html(finance_results, weather_result, seismic_result, dynamo_result)
    payload = json.dumps({
        "from":    from_email,
        "to":      [to_email],
        "subject": "FAWP Scanner Daily Digest",
        "html":    html,
    }).encode()
    req = urllib.request.Request(
        RESEND_API_URL,
        data=payload,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type":  "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception:
        return False
