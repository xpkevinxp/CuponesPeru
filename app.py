import argparse
import asyncio
import json
import os
from typing import List, Dict, Any

from bs4 import BeautifulSoup
from openai import OpenAI
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError


DEEPSEEK_TOKEN = os.environ.get("DEEPSEEK_API_KEY", "sk-f0a3bf1e44554bc6bea9936ea410db80")
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

LOAD_MORE_KEYWORDS = [
    "load more",
    "show more",
    "ver más",
    "ver mas",
    "mostrar más",
    "mostrar mas",
    "ver todo",
    "más resultados",
    "mas resultados",
    "load",  # fallback
]

COUPON_KEYWORDS = [
    "cupón",
    "cupon",
    "descuento",
    "promo",
    "promoción",
    "promocion",
    "oferta",
    "código",
    "codigo",
]


async def collect_dynamic_content(url: str, max_scrolls: int = 8, max_clicks: int = 5) -> Dict[str, Any]:
    """Visit a page, attempt to reveal lazy content, and return the final HTML."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")

        previous_height = 0
        scroll_events = 0
        for _ in range(max_scrolls):
            await page.mouse.wheel(0, 2400)
            await page.wait_for_timeout(1200)
            new_height = await page.evaluate("document.body.scrollHeight")
            scroll_events += 1
            if new_height == previous_height:
                break
            previous_height = new_height

        clicks = []
        for keyword in LOAD_MORE_KEYWORDS:
            locator = page.get_by_text(keyword, exact=False)
            count = await locator.count()
            for index in range(min(count, max_clicks)):
                try:
                    await locator.nth(index).click(timeout=2000)
                    await page.wait_for_timeout(1500)
                    clicks.append({"keyword": keyword, "index": index})
                except PlaywrightTimeoutError:
                    continue

        html = await page.content()
        await browser.close()
        return {"html": html, "scroll_events": scroll_events, "clicks": clicks}


def extract_coupon_candidates(html: str, limit: int = 30) -> List[str]:
    """Pull coupon-like snippets from the HTML to help the language model."""
    soup = BeautifulSoup(html, "html.parser")
    snippets: List[str] = []
    for element in soup.find_all(text=True):
        text = " ".join(element.strip().split())
        if not text:
            continue
        lower = text.lower()
        if any(keyword in lower for keyword in COUPON_KEYWORDS):
            snippet = element.parent.get_text(" ", strip=True)
            if snippet and snippet not in snippets:
                snippets.append(snippet)
        if len(snippets) >= limit:
            break
    return snippets


def summarize_with_ai(url: str, html: str, candidates: List[str]) -> List[Dict[str, Any]]:
    client = OpenAI(api_key=DEEPSEEK_TOKEN, base_url=DEEPSEEK_BASE_URL)
    condensed_html = html[:6000]
    joined_candidates = "\n".join(f"- {text}" for text in candidates) or "- (sin candidatos claros)"
    system = "Eres un asistente que extrae cupones y descuentos desde HTML."  # noqa: E501
    user = f"""
URL objetivo: {url}
HTML (recortado): {condensed_html}
Posibles textos relevantes:\n{joined_candidates}

Devuelve un JSON con una lista de objetos de cupones. Cada objeto debe incluir los campos:
- title: nombre corto del cupón o promoción
- description: detalles útiles del beneficio
- code: el código del cupón si existe, de lo contrario null
- value: porcentaje o monto del descuento si aparece, de lo contrario null
- link: URL objetivo (usa {url} si no hay una específica)

Si no hay cupones, responde [] como JSON sin texto adicional.
"""
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
    )
    raw_content = response.choices[0].message.content or "[]"
    try:
        parsed = json.loads(raw_content)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return [{"title": "No se pudieron interpretar cupones", "description": raw_content.strip(), "code": None, "value": None, "link": url}]


async def run(url: str) -> Dict[str, Any]:
    dynamic_result = await collect_dynamic_content(url)
    html = dynamic_result["html"]
    candidates = extract_coupon_candidates(html)
    coupons = summarize_with_ai(url, html, candidates)
    return {
        "url": url,
        "interactions": {
            "scroll_events": dynamic_result["scroll_events"],
            "clicks": dynamic_result["clicks"],
        },
        "candidates_found": len(candidates),
        "coupons": coupons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scraper de cupones con IA")
    parser.add_argument("url", help="Página a inspeccionar")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(run(args.url))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
