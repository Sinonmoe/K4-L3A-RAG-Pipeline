# RAG evaluation results

## Run information

| Field | Value |
|---|---|
| Evaluation date | 2026-09-20 |
| Framework and version | RAGAS 0.4.3 (`Faithfulness`, `ResponseRelevancy`, `LLMContextRecall`, `LLMContextPrecisionWithReference`); script: `group_project/evaluation/evaluate_ab.py`, raw per-case results: `group_project/evaluation/ab_results.json` |
| Evaluator model | `gpt-4o-mini` (OpenAI, temperature 0) as LLM judge; `text-embedding-3-small` for `ResponseRelevancy` |
| Generator model | `gpt-4o-mini` (OpenAI) via `task10_generation.call_llm`, temperature 0.3, top_p 0.9, same system prompt and citation validator for both configs |
| Embedding model | `text-embedding-3-small` (OpenAI, 1536 dim), Chroma cosine index |
| Corpus version/commit | `29fa48d` (last commit changing `data/standardized/`); 8 standardized documents, 283 chunks (`RecursiveCharacterTextSplitter`, size 500, overlap 50). Code base `bfa3653` plus the citation-normalization fix in the same commit as this report. |
| Golden dataset size | 15 grounded question-answer pairs |
| `top_k` | 5 |
| Fallback threshold and calibration | Production threshold = 0.50. Re-measured in this run with `text-embedding-3-small`: in-domain `IELTS writing task 2 band descriptors criteria` = 0.825, out-of-domain `công thức nấu phở bò Hà Nội gia truyền` = 0.227, so 0.50 sits between the two. The threshold is calibrated only on this corpus and embedding model. The PageIndex fallback was not included in the A/B because no `PAGEINDEX_API_KEY` was configured; both configs measure `retrieve()` without fallback. |

Run notes:
- The first run of this evaluation scored faithfulness 0.067 (A) and 0.000 (B), because 29 of 30 answers became the safe refusal. `gpt-4o-mini` copied the context label (`[ID: news/article_01.md::chunk-4 | Title: ...]`) into its citation, and `_citations_match_sources` rejected it. This was a pipeline bug (the chatbot would have refused almost every question), not a retrieval-quality result. `_normalize_citations` now strips the `ID:` label and the trailing metadata, with a regression test (`test_normalize_citations_strips_copied_context_label`). All numbers below are from the run after the fix.
- Single run, 15 cases, generator temperature 0.3. One case changing from correct to refused moves a metric by about 0.07, so differences of that size are within noise.
- RAGAS logged "LLM returned 1 generations instead of requested 3" for `ResponseRelevancy`, so each answer-relevance score is based on one generated question instead of three.

## Configurations

- **Config A — dense-only:** embed the question with `text-embedding-3-small`, query Chroma (cosine) and take the five highest-similarity chunks.
- **Config B — hybrid + RRF:** take the top 10 dense results and the top 10 BM25 results from the same 283 chunks, fuse once with RRF (`k=60`) and keep the top five.

Both configurations use the same 15 questions, chunks, generator, prompt, evaluator and `top_k`. Only retrieval strategy changes.

Metric definitions (RAGAS): faithfulness is the share of claims in the answer supported by the retrieved context; answer relevance is how well the answer addresses the question (embedding similarity to questions regenerated from the answer); context recall is how much of the reference answer (`expected_answer`) is supported by the retrieved context; context precision is the LLM-judged share of retrieved chunks that are useful for the reference answer, weighted by rank.

## Overall scores

| Metric | Config A | Config B | Delta B−A |
|---|---:|---:|---:|
| Faithfulness | 0.669 | 0.739 | +0.070 |
| Answer relevance | 0.852 | 0.806 | −0.046 |
| Context recall | 0.900 | 0.867 | −0.033 |
| Context precision | 0.736 | 0.729 | −0.008 |
| **Average** | **0.789** | **0.785** | **−0.004** |

Safe refusals: 1 of 15 answers in Config A and 2 of 15 in Config B. All three come from citation validation (wrong or missing citation), not from the model judging the evidence insufficient. No metric returned NaN.

## A/B comparison

- Cấu hình tốt hơn: **không có cấu hình thắng rõ ràng.** A cao hơn về relevance, recall, precision; B cao hơn về faithfulness; average gần như bằng nhau (0.789 so với 0.785).
- Evidence: khác biệt faithfulness của B đến từ các case 1, 9, 10, 11, 13 (B tốt hơn), trừ đi các case 3 và 12 (B kém hơn). Case 11 cho thấy lợi ích thật của hybrid: dense bỏ sót hai chunk chứa "bullet points"/"note form" (`examiner_comments.md::chunk-2`, `sample_tasks_2023.md::chunk-12`) còn BM25 tìm được, nên recall của case này tăng từ 0.0 lên 0.5. Case 3 cho thấy mặt trái: chunk chứa đáp án 25% (`article_02.md::chunk-5`) nằm ở hạng 5 của dense nhưng RRF đẩy nó ra khỏi top five, câu trả lời không có citation và bị từ chối. Với 15 case và một lần chạy, chênh lệch ≤ 0.07 không đủ để kết luận RRF cải thiện; chưa nên chọn B chỉ vì faithfulness cao hơn.
- Trade-off về latency/cost: retrieval 15 câu mất 7.63 giây (A) và 6.87 giây (B); thời gian bị chi phối bởi lời gọi embedding qua mạng nên chênh lệch này là nhiễu, BM25 và RRF chạy local không đáng kể. Generation mất 21.84 giây (A) và 22.61 giây (B). Token đầu vào cho generator là 12,412 (A) và 12,768 (B), tăng 2.9%; token đầu ra 1,064 và 1,046. Chi phí generator ước tính theo giá niêm yết `gpt-4o-mini` ($0.15/$0.60 mỗi 1M token vào/ra) là $0.00250 (A) và $0.00254 (B). Chi phí của evaluator RAGAS không được đo. Kết luận tạm thời: giữ B làm mặc định vì chi phí thêm gần bằng 0 và B phục hồi được truy vấn từ khóa như case 11, nhưng chỉ sau khi xử lý lỗi RRF đẩy chunk đúng ra ngoài top five (khuyến nghị 2).

## Worst performers

Xếp theo điểm trung bình bốn metric của từng cặp (case, config).

| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | Are bullet points or note form appropriate in IELTS Writing responses? | A | 0.000 | 0.920 | 0.000 | 0.000 | retrieval | Dense không lấy được hai chunk chứa đáp án (`examiner_comments.md::chunk-2`, `sample_tasks_2023.md::chunk-12`); top five toàn chunk không nhắc bullet points. Model trả lời "Không" (đúng) từ kiến thức nền và trích dẫn `article_01.md::chunk-5` không chứa ý đó, nên faithfulness bằng 0. Config B tìm được cả hai chunk nhờ BM25 (faithfulness 1.0, recall 0.5). |
| 2 | How much does each assessment criterion contribute to the IELTS essay score? | B | 0.000 | 0.000 | 0.000 | 1.000 | retrieval | Chunk đáp án `article_02.md::chunk-5` (25%) ở hạng 5 của dense bị RRF loại khỏi top five, thay bằng `article_02.md::chunk-1`. Model trả lời không có citation nên bị validator từ chối. Recall bằng 0 vì không chunk nào trong top five chứa con số 25 percent. Precision vẫn bằng 1.0 vì RAGAS chấm chunk theo mức liên quan đến câu hỏi, không theo việc chunk có chứa đáp án: cả năm chunk đều nói về tiêu chí chấm điểm IELTS nên đều bị coi là liên quan. Vì vậy không dùng precision riêng lẻ để kết luận có đáp án hay không; cần đọc cùng recall. Config A xử lý đúng case này (1.0/0.81/1.0/0.70). |
| 3 | How should the 60 minutes for the IELTS Writing test generally be divided between Task 1 and Task 2? | B | 0.000 | 0.000 | 1.000 | 0.917 | generation | Retrieval đúng (recall 1.0, chunk `article_02.md::chunk-5` có "20 minutes" và "40 minutes") nhưng model trích dẫn `[ID: article_02.md::chunk-5 | URL: ...]` thiếu tiền tố `news/`, validator từ chối và trả về câu từ chối. Config A cũng lỗi ở cùng case: trích dẫn `news/article_03.md::chunk-2` là chunk không có trong context (score trung bình 0.50). |

Ngoài ba case trên: case 2 (`What criterion replaces Task Achievement ...`) có câu trả lời đúng ("Task Response"), trích dẫn đúng chunk chứa cụm này, nhưng faithfulness vẫn bằng 0.0 ở cả hai config. Nguyên nhân chưa được xác minh; một giả thuyết là judge chấm câu trả lời tiếng Việt trên context tiếng Anh.

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
|---:|---|---|---|---|
| 1 | Làm citation chịu lỗi tốt hơn: mở rộng `_normalize_citations` để khớp theo hậu tố đường dẫn (`article_02.md::chunk-5` → `news/article_02.md::chunk-5`) và thêm ví dụ citation đúng định dạng vào prompt (hoặc dùng structured output trả về danh sách ID). | Lần chạy đầu 29/30 câu bị từ chối vì citation chép cả nhãn `ID:`; sau khi sửa vẫn còn 3/30 từ chối do citation thiếu tiền tố thư mục, sai chunk hoặc không có citation (worst case 3, cả hai config). | Giảm số câu từ chối do citation từ 3 xuống gần 0; ước tính faithfulness tăng khoảng 0.1 vì hai câu trả lời có bằng chứng đúng không còn bị loại. | Thêm unit test cho citation thiếu tiền tố; chạy lại `python -m group_project.evaluation.evaluate_ab` và đếm số answer bằng `SAFE_REFUSAL` (mục tiêu ≤ 1/30). |
| 2 | Sửa tầng fusion: lấy top 20 mỗi nhánh trước khi RRF (hiện 10) hoặc dùng RRF có trọng số ưu tiên dense; xem xét reranker (cross-encoder) sau RRF. | Case 3 (B): RRF đẩy chunk đáp án đang ở hạng 5 của dense ra ngoài top five. Case 11 (A): dense bỏ sót hai chunk chứa từ khóa nên BM25 là nhánh cứu được. Recall của B (0.867) thấp hơn A (0.900) dù hybrid đáng lẽ chỉ bổ sung. | Giữ được lợi ích của BM25 (case 11) mà không mất chunk dense đúng (case 3); kỳ vọng recall của B ≥ recall của A. | Chạy lại A/B trên cùng golden set; kiểm tra chunk `article_02.md::chunk-5` có trong top five của case 3 và `examiner_comments.md::chunk-2` có trong case 11; so sánh recall theo từng case. |
| 3 | Điều tra faithfulness = 0 ở câu trả lời đúng: ép generator trả lời cùng ngôn ngữ với câu hỏi (tiếng Anh cho câu hỏi tiếng Anh) và ghi lại các statement mà judge chấm. | Case 2 (cả hai config): câu trả lời đúng "Task Response", chunk trích dẫn có cụm này, recall 1.0, nhưng faithfulness 0.0. Toàn bộ câu trả lời hiện là tiếng Việt trên context tiếng Anh. | Loại bỏ nguồn nhiễu của metric để faithfulness phản ánh đúng chất lượng grounding. | Chạy lại riêng case 2 với prompt cùng ngôn ngữ và so faithfulness; nếu tăng lên thì lỗi nằm ở đo lường, không phải generation. |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
|---|---|---:|---:|---|
| Chưa thực hiện HyDE/reranker nâng cao | Hybrid + RRF | N/A | N/A | Chưa đủ điều kiện nhận bonus. Reranker được đề xuất ở khuyến nghị 2 và cần một phép đo A/B riêng với baseline RRF. |
