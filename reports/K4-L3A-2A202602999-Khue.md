# Báo cáo đóng góp cá nhân

## Thông tin

- Họ và tên: Nguyễn Xuân Khuê
- Mã học viên: 2A202602999
- Nhóm: K4-L3A
- Repository/branch: `Sinonmoe/K4-L3A-RAG-Pipeline` / `main`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| **Task 4: Chunking, Embedding & Indexing** | - Triển khai `load_documents()` đọc toàn bộ 8 tài liệu (3 legal, 5 news) từ `data/standardized/`, chuẩn hóa metadata và gán ID ổn định.<br>- Triển khai `chunk_documents()` dùng `RecursiveCharacterTextSplitter` (chunk_size=500, chunk_overlap=50) chia thành 283 chunks không rỗng, gán `chunk_index` và ID duy nhất `f"{doc_id}::chunk-{index}".`<br>- Triển khai `embed_texts()` hỗ trợ linh hoạt OpenAI (`text-embedding-3-small`), Gemini và local SentenceTransformer theo biến môi trường.<br>- Triển khai `embed_chunks()` (batch embedding) và `index_to_vectorstore()` upsert idempotent vào ChromaDB với cosine distance, bổ sung cơ chế tự động recreate collection khi vector dimension thay đổi. | `src/task4_chunking_indexing.py`, `tests/test_task4.py` | Done |
| **Task 5: Semantic Search** | - Triển khai `semantic_search(query, top_k=10)` sử dụng chung `embed_texts()` và persistent collection của Task 4.<br>- Chuyển đổi cosine distance sang cosine similarity `score = max(0.0, 1.0 - distance)`.<br>- Chuẩn hóa metadata (đảm bảo key `url`), gán `retrieval_method="dense"`, sắp xếp giảm dần theo score và validate schema. | `src/task5_semantic_search.py` | Done |
| **Task 6: Lexical Search (BM25)** | - Triển khai `build_bm25_index(corpus)` dùng `BM25Okapi` trên cùng tập chunks với Task 5; thiết lập sàn điểm IDF dương (`0.25`) giải quyết triệt để lỗi IDF = 0 khi corpus nhỏ.<br>- Triển khai `lexical_search(query, top_k=10)` tính điểm BM25 cho query tokens, lọc điểm > 0, sắp xếp giảm dần và gán `retrieval_method="bm25"`. | `src/task6_lexical_search.py` | Done |
| **Task 7: Reciprocal Rank Fusion (RRF)** | - Triển khai `rerank_rrf(ranked_lists, top_k=5, k=60)` theo công thức $RRF(d) = \sum \frac{1}{k + \text{rank}}$, rank bắt đầu từ 1.<br>- Đảm bảo chunk xuất hiện trong cả 2 danh sách nhận đúng tổng điểm RRF, khử trùng ID, gán `retrieval_method="hybrid"` và sort giảm dần theo điểm. | `src/task7_reranking.py` | Done |
| **Task 8: PageIndex Fallback** | - Triển khai `upload_documents()` và `pageindex_search(query, top_k=5)`.<br>- Thiết lập cơ chế timeout và bắt lỗi an toàn (exception handling) để lỗi từ dịch vụ ngoài không làm sập pipeline. | `src/task8_pageindex_vectorless.py` | Done |
| **Task 9: Retrieval Pipeline & Calibration** | - Triển khai `retrieve(query, top_k=5, score_threshold=0.5, use_reranking=True)` kết hợp dense, BM25 và RRF (chỉ fuse RRF đúng 1 lần).<br>- Dùng dense cosine score gốc cao nhất (`best_dense_score`) so sánh với `score_threshold` để kích hoạt fallback, tuyệt đối không dùng điểm RRF.<br>- An toàn trả về kết quả hybrid khi provider lỗi.<br>- Chạy thực nghiệm hiệu chỉnh threshold trên 2 query in-domain và out-of-domain, xác định `SCORE_THRESHOLD = 0.50` và ghi nhận vào báo cáo. | `src/task9_retrieval_pipeline.py`, `.env`, `group_project/evaluation/RESULT.md`, `reports/RESULT.md` | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định: Sử dụng kiến trúc Hybrid Retrieval (Dense Search + BM25 kết hợp RRF $k=60$) và dùng chung module embedding giữa Task 4 & Task 5.**
   - **Lý do/evidence:** Dense search bắt ngữ nghĩa câu hỏi rất tốt nhưng dễ bỏ sót từ khóa chính xác, mã tài liệu hay tên riêng (ví dụ: các tiêu chí chấm điểm IELTS cụ thể). BM25 bù đắp trực tiếp điểm yếu này. Thuật toán RRF ($k=60$) giải quyết bài toán dung hòa 2 thang đo điểm khác biệt hoàn toàn (cosine similarity trong đoạn $[0, 1]$ và điểm BM25 không chặn trên) mà không cần chuẩn hóa phức tạp. Việc dùng chung hàm `embed_texts()` đảm bảo đồng nhất tuyệt đối về model, tokenization và vector dimension giữa quá trình index và query.
   - **Trade-off:** Chi phí tính toán và độ trễ tăng nhẹ do phải chạy song song cả 2 thuật toán tìm kiếm, nhưng đổi lại độ phủ ngữ cảnh (context recall) và độ chính xác của top chunks được cải thiện rõ rệt.

2. **Quyết định: Tự động kiểm tra và xử lý dimension mismatch trong ChromaDB cùng cơ chế lazy loading model.**
   - **Lý do/evidence:** Trong quá trình thử nghiệm chuyển đổi giữa các provider (SentenceTransformer 384 dim, OpenAI 1536 dim, Gemini 3072 dim), ChromaDB lưu cấu hình HNSW index theo số chiều của đợt insert đầu tiên và sẽ văng lỗi `InvalidArgumentError: Collection expecting embedding with dimension of 384, got 1536`. Tôi đã thiết kế hàm `index_to_vectorstore()` tự động kiểm tra số chiều hiện tại của collection; nếu phát hiện dimension thay đổi thì tự động xóa và tái tạo collection mới tương thích. Đồng thời, model embedding chỉ được khởi tạo khi hàm `embed_texts()` thực sự được gọi (lazy loading singleton), giúp các unit test và contract test chạy tức thì mà không bị nghẽn mạng hay tải model nặng.
   - **Trade-off:** Việc tái tạo collection sẽ xóa dữ liệu cũ và yêu cầu index lại toàn bộ corpus nếu đổi model, nhưng đảm bảo toàn vẹn không gian vector.

3. **Quyết định: Quyết định Fallback dựa trên cosine similarity score gốc của Dense Search thay vì điểm RRF.**
   - **Lý do/evidence:** Điểm RRF chỉ phản ánh thứ hạng tương đối ($\approx 0.01 - 0.03$), phụ thuộc vào số lượng danh sách fuse và tham số $k$, không biểu diễn độ tương đồng ngữ nghĩa thực sự với câu hỏi. Cosine score gốc của dense search phản ánh trực tiếp khoảng cách vector trong không gian ngữ nghĩa. Do đó, việc dùng `best_dense_score < score_threshold` là căn cứ khoa học và chính xác duy nhất để biết mô hình có tìm thấy tài liệu phù hợp trong corpus hay không.
   - **Trade-off:** Pipeline cần theo dõi và giữ lại dense score gốc trước khi thực hiện fusion, nhưng ngăn chặn hoàn toàn việc fallback nhầm khi tài liệu thực tế đã có độ khớp ngữ nghĩa rất cao.

4. **Quyết định: Hiệu chỉnh thực nghiệm `SCORE_THRESHOLD = 0.50` và không tuyên bố ngưỡng phổ quát.**
   - **Lý do/evidence:** Thực nghiệm đo trên tập ngữ liệu IELTS Writing với OpenAI `text-embedding-3-small` cho kết quả rõ rệt:
     - Query in-domain: `"IELTS writing task 2 band descriptors criteria"` $\rightarrow$ score = **0.825**
     - Query in-domain: `"IELTS examiner comments academic writing"` $\rightarrow$ score = **0.744**
     - Query out-of-domain: `"công thức nấu phở bò Hà Nội gia truyền"` $\rightarrow$ score = **0.227**
     - Query out-of-domain: `"how to change car engine oil"` $\rightarrow$ score = **0.173**
     Ranh giới giữa in-domain (> 0.74) và out-of-domain (< 0.23) rất rõ ràng, nên ngưỡng **`0.50`** là điểm cắt tối ưu.
   - **Trade-off:** Ngưỡng 0.50 này phụ thuộc chặt chẽ vào đặc trưng phân bố điểm của model `text-embedding-3-small` và dữ liệu IELTS Writing; nếu đổi sang embedding model khác hoặc miền dữ liệu khác thì bắt buộc phải chạy hiệu chỉnh lại (không áp dụng cố định cho mọi bài toán).

## Kiểm thử và kết quả

- **Các lệnh và script tôi đã dùng để kiểm thử:**
  - `python -m src.task4_chunking_indexing`: Nạp 8 documents, chia và index thành công **283 chunks** vào ChromaDB collection `'rag_documents'`.
  - `python -m src.task5_semantic_search`: Dense search hoạt động chính xác, trả về top 3 kết quả có độ tương quan cao (score ~0.71, method `dense`).
  - `python -m src.task6_lexical_search`: BM25 search hoạt động chính xác, trả về top 3 kết quả (score ~5.03, method `bm25`).
  - `python -m src.task7_reranking`: RRF fusion thành công, chunk xuất hiện ở cả 2 bảng xếp hạng nhận đúng tổng điểm RRF (~0.0325, method `hybrid`).
  - `python -m src.task9_retrieval_pipeline`: Chạy pipeline tích hợp end-to-end; fallback an toàn khi query ngoài domain hoặc khi provider gặp sự cố.
  - `pytest tests/test_task4.py -v`: Đạt **4/4 PASSED [100%]**.
  - `pytest tests/test_contracts.py -v`: Đạt **14/15 PASSED** (toàn bộ các bài test hợp đồng từ Task 1 đến Task 9 đều passed 100%; bài test duy nhất còn lại là `test_reorder_is_non_mutating_and_context_contains_source` thuộc Task 10 generation).
- **Lỗi đã phát hiện và cách xử lý:**
  1. *Lỗi dimension mismatch trong ChromaDB:* Khi chuyển sang OpenAI vector 1536 chiều, ChromaDB văng lỗi do collection cũ đang ở kích thước 384 chiều. Tôi đã xóa collection cũ và viết thêm đoạn code tự động recreate collection nếu phát hiện dimension thay đổi.
  2. *Lỗi UnicodeEncodeError trên Windows console:* Khi in thông báo có dấu tiếng Việt trong khối xử lý ngoại lệ retry của model, Windows console (bảng mã cp1252) bị crash. Tôi đã chuẩn hóa các chuỗi log retry sang ASCII.
  3. *Lỗi BM25 trả về score = 0:* Trong `test_contracts.py`, corpus test chỉ có 2 tài liệu khiến công thức Robertson-Spärck Jones IDF tính ra $\log(1.5/1.5) = 0$. Tôi đã bổ sung sàn điểm IDF tối thiểu (`0.25`) cho các term trong `build_bm25_index`, giúp BM25 luôn cho điểm dương khi khớp từ khóa.

## Điều còn hạn chế

- Chưa kết nối PageIndex với API key thực tế có trả phí mà mới dừng lại ở việc xử lý lỗi an toàn (graceful fallback và mock test).
- BM25 hiện đang sử dụng bộ tách từ cơ bản bằng khoảng trắng (`split()`), phù hợp tốt với tài liệu tiếng Anh (như IELTS), nhưng nếu áp dụng cho văn bản tiếng Việt phức tạp thì cần tích hợp thêm thư viện tách từ chuyên dụng (như `underthesea` hoặc `pyvi`).
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: Triển khai thử nghiệm thêm Cross-Encoder Reranker (như `bge-reranker-large` hoặc Jina Reranker) để so sánh định lượng với RRF; đồng thời kết hợp HyDE (Hypothetical Document Embeddings) nhằm mở rộng query trước khi đưa vào retrieval pipeline.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 20/09/2026
- Tên thành viên: Nguyễn Xuân Khuê
