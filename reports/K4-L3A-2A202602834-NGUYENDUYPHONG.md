# Báo cáo đóng góp cá nhân

## Thông tin

- Họ và tên: Duy Phong *(cần bổ sung họ tên đầy đủ)*
- Mã học viên: MSSV *(cần thay bằng mã học viên)*
- Nhóm: K4-L3A
- Repository/branch: `Sinonmoe/K4-L3A-RAG-Pipeline` / `main`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Golden dataset | Xây dựng 15 trường hợp hỏi–đáp grounded cho miền IELTS Writing, mỗi mẫu có câu hỏi, đáp án và ngữ cảnh kỳ vọng. | `group_project/evaluation/golden_dataset.json` | Done |
| A/B evaluation | Xây dựng evaluator offline tái lập được, so sánh dense-only với hybrid BM25 + RRF, lưu điểm tổng hợp và kết quả từng câu. | `group_project/evaluation/evaluate_ab.py`, `ab_results.json`, `RESULT.md` | Done |
| Generation và citation | Hoàn thiện reorder context, định dạng citation, gọi OpenAI/Gemini/Anthropic và safe refusal khi thiếu nguồn hoặc provider lỗi. | `src/task10_generation.py` | Done |
| Chatbot UI | Kết nối Streamlit với RAG pipeline, lưu lịch sử hội thoại và hiển thị nguồn, retrieval method, score. | `app.py` | Done |
| Kiểm thử nghiệm thu | Chạy contract và acceptance tests, xử lý các phần còn thiếu cho đến khi toàn bộ test đạt. | `tests/test_contracts.py`, `tests/test_acceptance.py` | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Chỉ đưa vào golden dataset các câu hỏi có thể đối chiếu trực tiếp với corpus chuẩn hóa.
   **Lý do/evidence:** Cả 15 mẫu đều chỉ rõ tài liệu trong `expected_context`, giúp đo retrieval thay vì dựa vào kiến thức có sẵn của mô hình.
   **Trade-off:** Dữ liệu dễ kiểm chứng nhưng mới tập trung vào factual QA tiếng Anh trong miền IELTS Writing.

2. **Quyết định:** Khi môi trường không có API key và Windows chặn DLL của `sentence-transformers`, sử dụng evaluator proxy offline và công khai phương pháp thay vì điền điểm ước lượng.
   **Lý do/evidence:** Script dùng cùng corpus, golden set, generator và `top_k=5` cho hai cấu hình; raw results được lưu trong `ab_results.json`. Hybrid + RRF đạt 0.582 so với 0.570 của dense-only.
   **Trade-off:** Kết quả tái lập và không tốn API nhưng faithfulness của extractive generator bằng 1.000 theo thiết kế, nên chưa thay thế được RAGAS/LLM judge.

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng: `python -m group_project.evaluation.evaluate_ab`, `python -m pytest -q` và `python -m py_compile src/task10_generation.py group_project/evaluation/evaluate_ab.py app.py`.
- Kết quả trước/sau nếu có: golden dataset từ file rỗng thành 15 mẫu hợp lệ; acceptance từ lỗi do dataset/report chưa hoàn thành thành toàn bộ `20 passed`. A/B cho thấy hybrid tăng average 0.012, answer relevance 0.028 và context recall 0.021.
- Lỗi đã phát hiện và cách xử lý: thiếu implementation ở generation/UI được bổ sung; thiếu khóa LLM được xử lý bằng safe refusal; lỗi DLL của local embedding được cô lập bằng evaluator TF-IDF offline và ghi rõ giới hạn trong báo cáo.

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm: evaluation hiện là proxy offline, chưa có LLM judge; golden dataset chưa có câu hỏi out-of-domain để đo safe refusal.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: cấu hình API, chạy lại RAGAS 0.4.3 trên cùng 15 câu, bổ sung câu hỏi paraphrase/out-of-domain và làm sạch navigation/footer trước khi re-index.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 20/09/2026
- Tên thành viên: Duy Phong *(cần xác nhận họ tên đầy đủ)*
