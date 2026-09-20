"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium
    
-> Dùng Firecrawl or bất cứ công cụ nào bạn quen    
"""

import asyncio
import io
import json
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from markitdown import MarkItDown


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Chủ đề: IELTS Writing — band descriptors, tiêu chí chấm điểm, bài viết mẫu
ARTICLE_URLS = [
    "https://ielts.idp.com/results/scores/writing",
    "https://ielts.org/news-and-insights/10-steps-to-writing-high-scoring-ielts-essays",
    "https://ieltsliz.com/ielts-writing-task-2/",
    "https://ieltsliz.com/ielts-sample-essay/",
    "https://ieltsliz.com/ielts-solution-essay-band-9-model-answer/",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
NOISE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "iframe", "svg"]
CONTENT_SELECTORS = ["article", "main", ".entry-content", "[role=main]"]


def _fetch_and_convert(url: str) -> dict:
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else ""
    if not title and soup.h1:
        title = soup.h1.get_text(strip=True)

    for tag in soup(NOISE_TAGS):
        tag.decompose()
    container = next((el for sel in CONTENT_SELECTORS if (el := soup.select_one(sel))), soup.body)

    converted = MarkItDown().convert_stream(
        io.BytesIO(str(container).encode("utf-8")), file_extension=".html"
    )
    return {
        "url": url,
        "title": title or "Unknown",
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": converted.text_content.strip(),
    }


async def crawl_article(url: str) -> dict:
    return await asyncio.to_thread(_fetch_and_convert, url)


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            article = await crawl_article(url)
            output = DATA_DIR / f"article_{index:02d}.json"
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output}")
        except Exception as error:
            print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
