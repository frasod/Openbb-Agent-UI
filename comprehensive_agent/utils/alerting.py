import smtplib
from typing import List, Dict
from email.message import EmailMessage

from ..config import settings

def send_alert(search_results: List[Dict], threshold: float = 0.5) -> bool:
    high_impact = [r for r in search_results if abs(r['sentiment_score']) > threshold]
    if not high_impact:
        return False
    msg = EmailMessage()
    msg.set_content(f'High impact news detected: {len(high_impact)} items')
    msg['Subject'] = 'Financial Alert'
    msg['From'] = settings.email_from
    msg['To'] = settings.email_to
    try:
        with smtplib.SMTP(settings.smtp_server) as server:
            server.login(settings.smtp_user, settings.smtp_pass)
            server.send_message(msg)
        return True
    except Exception:
        return False 