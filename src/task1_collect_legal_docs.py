"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Hướng dẫn:
    1. Chọn chủ đề của nhóm.
    2. Tìm tối thiểu 3 tài liệu PDF/DOCX từ nguồn công khai.
    3. Lưu file gốc vào data/landing/legal/.
    4. Đặt tên không dấu và thể hiện đúng nội dung.

Ví dụ tài liệu: học phí, học bổng, ký túc xá, quy trình đăng ký.
Nếu website chặn crawler, hãy chọn nguồn công khai khác; không vượt WAF.
"""

from pathlib import Path

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# Chủ đề: IELTS Writing — tài liệu chính thức từ ielts.org
SOURCES = {
    "ielts_writing_band_descriptors.pdf": "https://ielts.org/cdn/Guides/ielts-writing-band-descriptors.pdf",
    "ielts_academic_writing_sample_tasks_2023.pdf": "https://ielts.org/cdn/Sample-tests/ielts-academic-writing-sample-tasks-2023.pdf",
    "ielts_academic_writing_example_responses_examiner_comments.pdf": "https://ielts.org/cdn/computer-delivered-sample-tests-academic-writing/ielts-academic-writing-example-responses-to-parts-1-and-2-with-band-scores-and-examiner-comments.pdf",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Tải ít nhất 3 PDF/DOCX từ nguồn công khai."""
    for filename, url in SOURCES.items():
        target = DATA_DIR / filename
        if target.exists() and target.stat().st_size > 1024:
            print(f"Skip (exists): {target.name}")
            continue
        try:
            response = requests.get(url, headers=HEADERS, timeout=60)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise ValueError("response is not a PDF")
            target.write_bytes(response.content)
            print(f"Saved: {target.name} ({len(response.content)} bytes)")
        except Exception as error:
            print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    setup_directory()
    download_documents()
