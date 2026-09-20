# Báo cáo đóng góp cá nhân

## Thông tin

- Họ và tên: Duy Phong *(cần bổ sung họ tên đầy đủ)*
- Mã học viên: *(cần bổ sung)*
- Nhóm: K4-L3A
- Repository/branch: `Sinonmoe/K4-L3A-RAG-Pipeline` / `main`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Golden dataset | Xây dựng 15 trường hợp hỏi–đáp có đáp án kỳ vọng và ngữ cảnh đối chiếu cho miền IELTS Writing. | `group_project/evaluation/golden_dataset.json` (working tree của `duyphong134`) | Done |
| Kiểm tra acceptance | Kiểm tra cú pháp JSON và chạy test xác nhận dataset đủ số lượng, đúng schema, không có trường rỗng. | `tests/test_acceptance.py::test_golden_dataset_has_15_grounded_cases` | Done |
| Rà soát nghiệm thu | Chạy toàn bộ acceptance suite và xác định lỗi còn lại nằm ở báo cáo evaluation chưa hoàn tất, không phải golden dataset. | `tests/test_acceptance.py`, `group_project/evaluation/RESULT.md` | Partial |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Chỉ tạo câu hỏi có thể trả lời trực tiếp từ corpus đã chuẩn hóa thay vì dùng câu hỏi kiến thức IELTS chung.
   **Lý do/evidence:** Mỗi mẫu đều có `expected_context` chỉ rõ tài liệu nguồn, gồm các file `article_01.md`, `article_02.md`, `article_05.md` và hai tài liệu IELTS chính thức trong `data/standardized/legal/`. Cách này giúp đánh giá được khả năng truy xuất đúng ngữ cảnh, không chỉ kiểm tra kiến thức có sẵn của mô hình.
   **Trade-off:** Bộ dữ liệu bám sát corpus và dễ kiểm chứng nhưng phạm vi chủ đề còn hẹp, chủ yếu tập trung vào IELTS Writing.

2. **Quyết định:** Giữ câu trả lời kỳ vọng ngắn, đơn nghĩa và đưa tên nguồn vào `expected_context`.
   **Lý do/evidence:** Các câu hỏi tập trung vào dữ kiện rõ ràng như bốn tiêu chí chấm điểm, giới hạn từ, phân bổ thời gian và cấu trúc bài viết; điều này giảm mơ hồ khi chấm answer relevance và context recall.
   **Trade-off:** Câu trả lời ngắn thuận lợi cho đánh giá tự động nhưng chưa kiểm thử sâu các câu hỏi tổng hợp nhiều đoạn hoặc suy luận phức tạp.

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng: `python -m json.tool group_project/evaluation/golden_dataset.json` và `python -m pytest tests/test_acceptance.py -q`.
- Kết quả: JSON hợp lệ; test riêng `test_golden_dataset_has_15_grounded_cases` đạt `1 passed`. Toàn bộ acceptance suite đạt `4 passed, 1 failed`.
- Lỗi đã phát hiện và cách xử lý: file golden dataset ban đầu rỗng nên không thỏa điều kiện tối thiểu 15 mẫu. Tôi bổ sung 15 object có đủ `question`, `expected_answer`, `expected_context`. Lỗi acceptance còn lại do `group_project/evaluation/RESULT.md` vẫn chứa `TODO` và cần kết quả evaluation thực tế để hoàn thiện.

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm: dataset mới kiểm tra factual QA bằng tiếng Anh và chưa có câu hỏi ngoài miền để đo safe refusal hoặc các trường hợp diễn đạt lại khó.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: bổ sung nhãn ID/nguồn có cấu trúc, câu hỏi paraphrase và out-of-domain; sau đó chạy A/B dense-only so với hybrid + RRF để ghi kết quả thực đo vào báo cáo evaluation.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 20/09/2026
- Tên thành viên: Duy Phong *(cần xác nhận họ tên đầy đủ)*
