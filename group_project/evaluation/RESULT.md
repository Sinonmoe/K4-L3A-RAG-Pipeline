# RAG evaluation results

## Run information

| Field | Value |
|---|---|
| Evaluation date | 2026-09-20 |
| Framework and version | Offline proxy evaluator v1 (`scikit-learn` TF-IDF + token coverage); script: `group_project/evaluation/evaluate_ab.py` |
| Evaluator model | Deterministic local metrics, no LLM judge |
| Generator model | Deterministic extractive top-sentence generator, identical for both configs |
| Embedding model | Character TF-IDF, word-boundary n-grams 3–5 (offline evaluation index) |
| Corpus version/commit | `395946a`; 8 standardized documents, 283 chunks |
| Golden dataset size | 15 grounded question-answer pairs |
| `top_k` | 5 |
| Fallback threshold and calibration | Production threshold = 0.50. Previous calibration used in-domain `IELTS writing task 2 band descriptors criteria` (~0.825) and out-of-domain `công thức nấu phở bò Hà Nội gia truyền` (~0.227). Fallback was not included in this A/B because no `PAGEINDEX_API_KEY` was configured. |

The environment had no LLM/API keys, and the installed `sentence-transformers` path was blocked by Windows Application Control while loading a `pandas` DLL. Therefore this run uses an explicitly labelled, reproducible offline proxy instead of reporting fabricated RAGAS/LLM-judge scores. Raw per-case results are stored in `group_project/evaluation/ab_results.json`.

## Configurations

- **Config A — dense-only:** character TF-IDF retrieval over the same 283 chunks; take the five highest cosine-similarity results.
- **Config B — hybrid + RRF:** retrieve top 10 from the same TF-IDF index and top 10 from project BM25, fuse once with RRF (`k=60`), then keep top five.

Both configurations use the same 15 questions, chunks, extractive generator, evaluator, and `top_k`. Only retrieval strategy changes.

Metric definitions: faithfulness is the fraction of generated-answer tokens present in retrieved context; answer relevance is TF-IDF cosine similarity between extracted and expected answers; context recall is expected-answer token coverage by retrieved context; context precision is the fraction of retrieved chunks from the expected source document.

## Overall scores

| Metric | Config A | Config B | Delta B−A |
|---|---:|---:|---:|
| Faithfulness | 1.000 | 1.000 | +0.000 |
| Answer relevance | 0.137 | 0.165 | +0.028 |
| Context recall | 0.768 | 0.789 | +0.021 |
| Context precision | 0.373 | 0.373 | +0.000 |
| **Average** | **0.570** | **0.582** | **+0.012** |

Faithfulness reaches 1.000 by construction because the offline generator copies one retrieved sentence. This score verifies grounding but does not measure synthesis quality and must not be compared directly with an LLM-judge faithfulness score.

## A/B comparison

- Cấu hình tốt hơn: **Config B — hybrid + RRF**, nhưng mức cải thiện nhỏ.
- Evidence: average tăng từ 0.570 lên 0.582; answer relevance tăng 0.028 và context recall tăng 0.021, trong khi context precision không đổi.
- Trade-off về latency/cost: Config A mất 0.132 giây cho 15 câu; Config B mất 0.261 giây, chậm hơn khoảng 0.129 giây (xấp xỉ 98%) do chạy thêm BM25 và RRF. Cả hai không phát sinh API cost trong phép đo offline.

## Worst performers

| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | What does coherence mean in the IELTS Writing assessment criteria? | B | 1.000 | 0.000 | 0.455 | 0.200 | generation | Extractor chọn câu nói chung về band descriptors thay vì câu định nghĩa coherence nằm trong context. |
| 2 | What structure is generally recommended for an IELTS Task 2 essay? | B | 1.000 | 0.057 | 0.706 | 0.000 | retrieval/generation | Top five không chứa đúng `article_02.md`; extractor chọn một tiêu đề liên kết thay vì nội dung cấu trúc bài. |
| 3 | What is the minimum word count for IELTS Writing Task 2? | B | 1.000 | 0.220 | 0.429 | 0.400 | generation/data | Câu trích xuất có số 250 nhưng kèm bình luận thừa; token recall thấp do expected answer quá ngắn và cách đo token nhạy với diễn đạt. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
|---:|---|---|---|---|
| 1 | Thay extractive top-sentence bằng generator có prompt citation và LLM judge khi có API key. | Faithfulness proxy luôn bằng 1.000 nhưng relevance chỉ 0.165; hai worst cases chọn tiêu đề/câu không trả lời trực tiếp. | Tăng answer relevance và đánh giá được synthesis thực tế. | Cấu hình API, chạy lại cùng 15 câu bằng RAGAS 0.4.3 và lưu cả raw answers/citations. |
| 2 | Làm sạch Markdown trước khi chunk: bỏ navigation, footer, image/link-only lines và chuẩn hóa lỗi mã hóa. | Retrieval chọn các tiêu đề liên kết như “Essay Structure & Paragraphing” và nhiều chunk `article_01.md` chứa nội dung điều hướng. | Tăng context precision, giảm nhiễu cho generator. | Re-index corpus rồi so sánh precision và ba worst cases với baseline hiện tại. |
| 3 | Bổ sung metadata/source-aware retrieval và tinh chỉnh chunk size/overlap. | Câu về cấu trúc bài không lấy được expected source trong top five; precision trung bình chỉ 0.373. | Đưa đúng tài liệu vào top-k và cải thiện recall/precision. | Grid search chunk size/overlap và đo Recall@5, MRR, bốn metric trên cùng golden set. |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
|---|---|---:|---:|---|
| Chưa thực hiện HyDE/reranker nâng cao | Hybrid + RRF | N/A | N/A | Không yêu cầu bonus khi pipeline cơ bản và LLM evaluation chưa được cấu hình đầy đủ. |
