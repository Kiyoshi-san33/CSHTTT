# Vehicle Material Prediction – Research Prototype (LLM + Hybrid RAG + CoVe)

Hệ thống thực nghiệm và đánh giá dự đoán vật liệu linh kiện ô tô dựa trên phương pháp của **Wagner et al. (2026)** (*Exploring LLMs and RAG for Plausible and Explainable Material Prediction of Vehicle Components*), được thiết kế tối ưu cho laptop **Windows 11 / Lenovo AMD 32 GB RAM (CPU-only)** kết hợp inference qua Cloud API.

---

## 1. Technology Stack & Quyết định Kiến trúc

| Layer | Công nghệ | Vai trò |
|---|---|---|
| **Language & Backend** | Python 3.11 | Điều phối LLM, RAG, thực nghiệm & lưu trữ |
| **Frontend / Research UI** | Streamlit (Multipage) | Dự đoán tương tác, theo dõi thực nghiệm, gán nhãn mù & báo cáo |
| **Main LLM A** | Groq – `openai/gpt-oss-120b` | Model thực nghiệm chính thức A |
| **Main LLM B** | Google – `gemini-3.8-flash` | Model thực nghiệm chính thức B |
| **Local Dev LLM** | Ollama – `qwen3:8b` (`--dev`) | Chỉ dùng phát triển & kiểm thử offline |
| **Supplementary Judge** | Groq – `qwen/qwen3.8-27b` | LLM-as-a-Judge độc lập (Preview, tách riêng khỏi đánh giá của người) |
| **Dense Retrieval** | `BAAI/bge-small-en-v1.5` + FAISS CPU | Truy xuất ngữ nghĩa (`IndexFlatIP`, 384 chiều) — profile `adapted` |
| **Reproduction Dense** | `facebook/dpr-*-multiset-base` + FAISS | Profile `reproduction` đối chiếu với cấu hình DPR của bài báo |
| **Sparse Retrieval** | `bm25s` + `PyStemmer` | Truy xuất từ khóa BM25 (k1=1.5, b=0.75, stopwords=en, Snowball stemmer) |
| **Hybrid Fusion** | Reciprocal Rank Fusion (RRF, k=60) | Gộp Top-50 BM25 + Top-50 Dense → chọn Top-10 passages |
| **Data Storage** | SQLite (`database/experiments.db`) + JSONL/CSV | Lưu cấu hình, prompt version, token usage, passages, nhãn & xếp hạng |
| **Testing** | Pytest | Kiểm thử đơn vị và kiểm thử tích hợp toàn tuyến (end-to-end smoke test) |

---

## 2. Đối chiếu với Bài báo gốc (Wagner et al., 2026) – Tái lập & Cải tiến

Dự án bảo toàn **100% phương pháp luận cốt lõi** của bài báo gốc, đồng thời bổ sung các **điều chỉnh & cải tiến (Adaptations & Extensions)** để tối ưu cho môi trường máy trạm đơn (Windows CPU-only, 32 GB RAM) và mở rộng khả năng đánh giá tự động.

### 2.1 Những thành phần THAY THẾ / ĐIỀU CHỈNH (Adaptations & Improvements)

| Hạng mục | Bài báo gốc (Wagner et al., 2026) | Dự án hiện tại (`config.yaml`) | Lý do thay thế & Ghi chú |
|---|---|---|---|
| **LLM cho Experiment I** *(Material-only baseline, 100 linh kiện)* | **Mixtral-8x22B** | **`openai/gpt-oss-120b`** (qua Groq API) | Mixtral-8x22B đã cũ/khó truy cập; thay bằng model mở 120B hiện đại, tốc độ cao trên Groq. |
| **2 LLM cho Experiment II** *(2 LLM × 3 phương pháp, 50 linh kiện)* | **Mixtral-8x22B** (Open-weight) & **GPT-4** (Proprietary) | **`openai/gpt-oss-120b`** (Groq) & **`gemini-3.8-flash`** (Google API) | Giữ nguyên cấu trúc **1 model trọng số mở lớn vs. 1 model thương mại**, nhưng tối ưu chi phí và tốc độ gọi API. |
| **Vector DB & Sparse Search Engine** | **OpenSearch** + **Haystack framework** (cần chạy cụm server Docker nặng RAM) | **`bm25s`** (Lucene BM25 + Snowball stemmer) + **FAISS CPU (`IndexFlatIP`)** | Không cần cài Docker/OpenSearch, chạy trực tiếp trên ổ cứng & RAM laptop 32 GB cực nhanh, giữ nguyên 100% công thức toán học Lucene BM25 (`k1=1.5, b=0.75`), Exact Inner Product và RRF (`k=60`). |
| **Dense Embedding Model (Retrieval)** | **Facebook DPR** (`dpr-ctx_encoder-multiset-base` & `dpr-question_encoder-multiset-base`, 768 chiều) | Hỗ trợ **cả 2 chế độ (`profile`)**:<br>• **`adapted` (Mặc định):** `BAAI/bge-small-en-v1.5` (384 chiều, 512 tokens)<br>• **`reproduction`:** Giữ nguyên cặp **Facebook DPR** của bài báo | `bge-small-en-v1.5` nhẹ hơn một nửa, hỗ trợ ngữ cảnh 512 tokens (khớp đoạn 200 từ tốt hơn giới hạn 256 của DPR) và chạy CPU nhanh gấp ~4 lần. Vẫn có thể bật `--profile reproduction` để chạy đúng DPR của tác giả. |
| **Định dạng đầu ra Prompt** | Văn bản tự do, tác giả tách vật liệu thủ công | Thêm 1 câu hậu tố (`enforce_list_format: true` trong `config.yaml`) yêu cầu liệt kê dạng `1. Material: explanation` | Giúp hệ thống tự động bóc tách chính xác tên vật liệu và lời giải thích mà không làm thay đổi nội dung suy luận. |
| **Đánh giá (Evaluation)** | Chỉ có **Chuyên gia con người** (4 người ở Exp I; 10 người chia nhóm 3 người/linh kiện ở Exp II) | **Giao diện gán nhãn mù cho Người** (`app/pages/annotation.py`) **+ Bổ sung Giám khảo LLM độc lập** (`qwen/qwen3.8-27b` trong `src/evaluation/llm_judge.py`) | Kết quả của Người (`kind="human"`) và LLM Judge (`kind="llm"`) được **lưu tách biệt hoàn toàn**, đồng thời có thêm báo cáo độ đồng thuận giữa Người và LLM Judge (`Human-LLM Agreement`). |

### 2.2 Những thành phần GIỮ NGUYÊN 100% theo Bài báo gốc (Core Reproduction)

1. **Tiêu chí lọc Wikipedia & Cắt đoạn (`src/rag/wiki/extract.py`, `src/rag/chunking.py`):**
   - Lọc bài viết Wikipedia tiếng Anh thỏa mãn: chứa **≥ 2 tên linh kiện đa từ** HOẶC (**≥ 3 từ khóa vật liệu** VÀ **≥ 3 từ khóa ô tô**).
   - Cắt bài viết thành các đoạn văn **200 từ**, độ chồng lấn (**overlap**) **50 từ**.
2. **Truy hồi lai Hybrid Retrieval (`src/rag/hybrid_retriever.py`):**
   - Lấy **Top-50** từ Sparse (BM25) + **Top-50** từ Dense → Gộp bằng thuật toán **Reciprocal Rank Fusion (RRF, $k=60$)** → Chọn ra **Top-10 passages** đưa vào prompt.
3. **3 Phương pháp trong Experiment II (`no_rag.py`, `rag.py`, `cove.py`):**
   - **No-RAG:** Hỏi LLM trực tiếp (Phụ lục B).
   - **RAG:** Truy hồi Top-10 đoạn văn Wikipedia cho linh kiện rồi đưa vào prompt (Phụ lục B).
   - **CoVe (Chain-of-Verification – 4 bước, Phụ lục C):** *(1) Baseline* → *(2) Plan verification questions* → *(3) Execute verification với RAG* → *(4) Final verified response*.
4. **Bộ chỉ số thống kê & đánh giá (`src/evaluation/`):**
   - **Experiment I:** `Coverage@1`, `Coverage@3`, `Precision@1`, `Precision@3` trên 100 linh kiện.
   - **Experiment II:** `Coverage@2`, `Precision@2`, tỷ lệ `Factual Error`, điểm giải thích `Likert 1–5`, kiểm định tương quan thứ hạng **Kendall's $\tau$** và độ đồng thuận **Krippendorff's $\alpha$** trên 50 linh kiện.

---

## 3. Cấu trúc Thư mục

```
vehicle_material_reproduction_project/
├── app/
│   ├── streamlit_app.py                # Điểm vào giao diện Streamlit đa trang
│   ├── pages/
│   │   ├── prediction.py               # Dự đoán thử nghiệm 1 linh kiện (No-RAG / RAG / CoVe)
│   │   ├── experiments.py              # Tiến độ thực nghiệm, tra cứu log & xuất JSONL/CSV
│   │   ├── annotation.py               # Giao diện gán nhãn mù cho Người (Exp I & Exp II)
│   │   └── evaluation.py               # Bảng kết quả, kiểm định thống kê & biểu đồ
│   └── components/
│       ├── passage_view.py
│       └── sidebar.py
├── src/
│   ├── core/                           # Đọc config, áp dụng retrieval profile, paths, prompt loader
│   ├── llm/                            # BaseLLM, OpenAICompatLLM, groq/gemini/ollama/dummy clients
│   ├── rag/                            # Wikipedia ingestion, chunking (200/50), BGE/DPR, BM25, FAISS, RRF
│   ├── experiments/                    # no_rag.py, rag.py, cove.py, material_only.py (Exp I), runner.py
│   ├── evaluation/                     # coverage, precision, factual_errors, human_ratings, agreement, report
│   ├── storage/                        # SQLAlchemy 2.0 models, SQLite WAL db, repository, export
│   └── schemas/                        # Pydantic schemas (prediction, experiment, annotation)
├── prompts/                            # Toàn bộ prompt dạng .txt (kèm mã băm phiên bản name@sha256[:8])
├── data/
│   ├── raw/wiki_dump/                  # Wikipedia multistream dump (.xml.bz2 + index)
│   ├── processed/components_filter.txt # Danh sách linh kiện đa từ dùng lọc bài viết
│   ├── corpus/articles.jsonl           # Bài viết Wikipedia sau khi lọc và làm sạch
│   ├── chunks/passages.jsonl           # Các đoạn văn 200 từ (chồng lấn 50 từ)
│   └── test_sets/                      # exp1_components.txt (100), exp2_components.txt (50)
├── indexes/
│   ├── bm25/                           # Chỉ mục bm25s + meta.json
│   └── faiss/{bge,dpr}/                # Chỉ mục FAISS IndexFlatIP + shards + meta.json
├── results/
│   ├── predictions/                    # File xuất JSONL/CSV các lượt chạy và lời gọi LLM
│   ├── annotations/                    # File xuất CSV nhãn đánh giá và xếp hạng
│   ├── metrics/                        # report.md, metrics.json, human_llm_disagreements.csv
│   └── plots/                          # fig_exp1.png, fig_exp2.png
├── database/experiments.db             # Cơ sở dữ liệu SQLite chính
├── tests/                              # Bộ kiểm thử Pytest
├── config.yaml
├── requirements.txt
└── .env.example
```

---

## 4. Cài đặt & Kích hoạt Môi trường

> **⚠️ LƯU Ý QUAN TRỌNG TRƯỚC KHI CHẠY BẤT KỲ LỆNH NÀO:**
> Mỗi khi mở một cửa sổ Terminal / PowerShell mới, bạn **BẮT BUỘC phải kích hoạt môi trường ảo `.venv`** (khi kích hoạt thành công, đầu dòng lệnh sẽ có chữ `(.venv)`).
> Nếu quên kích hoạt `.venv`, Windows sẽ dùng Python mặc định của máy và báo lỗi thiếu thư viện như `ModuleNotFoundError: No module named 'yaml'`.

### 4.1 Kích hoạt môi trường ảo `.venv` (Mỗi lần mở PowerShell mới)
```powershell
cd d:\vehicle_material_reproduction_project

# Nếu PowerShell báo lỗi chặn script (ExecutionPolicy), chạy lệnh dưới đây trước:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Kích hoạt môi trường ảo .venv:
.\.venv\Scripts\Activate.ps1
```
*(Mẹo: Nếu không muốn kích hoạt `.venv`, bạn có thể thay chữ `python` bằng `.\.venv\Scripts\python.exe` ở đầu mọi câu lệnh bên dưới).*

### 4.2 Tạo mới / Cài đặt lại thư viện (Chỉ chạy khi chưa có `.venv` hoặc thiếu gói)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4.3 Cấu hình API Key & Kiểm tra hệ thống
```powershell
# 1. Tạo file .env từ file mẫu (nếu chưa có) và điền GROQ_API_KEY, GEMINI_API_KEY
if (!(Test-Path .env)) { Copy-Item .env.example .env }

# 2. Chạy bộ kiểm thử tự động (offline, dùng dữ liệu giả + dummy LLM)
python -m pytest -v

# 3. Kiểm tra kết nối và danh sách model trên Groq / Gemini trước khi chạy thật
python -m src.llm.check --ping
```

---

## 5. Quy trình Thực nghiệm Đầy đủ (End-to-End CLI)

> **Nhắc lại:** Đảm bảo đầu dòng lệnh PowerShell đã có `(.venv)`. Nếu chưa, hãy chạy `.\.venv\Scripts\Activate.ps1`.

### Bước 1 – Chuẩn bị Kho ngữ liệu & Chỉ mục Hybrid RAG
```powershell
# 1.1 Tải Wikipedia dump tiếng Anh (~24 GB, hỗ trợ tự tải tiếp Resume nếu bị ngắt mạng)
python -m src.rag.ingestion download

# 1.2 Trích xuất & lọc bài viết liên quan đến ô tô/vật liệu (chạy đa nhân CPU)
python -m src.rag.ingestion extract

# 1.3 Cắt bài viết thành các đoạn văn (passage) 200 từ, chồng lấn (overlap) 50 từ
python -m src.rag.chunking

# 1.4 Xây dựng chỉ mục tìm kiếm BM25 + FAISS
#     - Profile mặc định (adapted = BAAI/bge-small-en-v1.5 + FAISS + bm25s):
python -m src.rag.indexing --all --profile adapted

#     - (Tùy chọn) Xây dựng thêm chỉ mục DPR cho profile reproduction:
python -m src.rag.indexing --dense --profile reproduction

# 1.5 Kiểm tra nhanh kết quả truy hồi lai (BM25 + Dense + RRF)
python -m src.rag.hybrid_retriever brake disc

# 1.6 Đánh giá nhanh chất lượng tầng Retrieval
python -m src.evaluation.retrieval_check
```

### Bước 2 – Chạy Thực nghiệm Dự đoán Vật liệu
```powershell
# 2.1 Experiment I – Adapted Baseline
#     (Model: GPT-OSS 120B, Few-shot Phụ lục A, dự đoán trên 100 linh kiện)
python -m src.experiments.runner --exp exp1

# 2.2 Experiment II – 6 cấu hình thực nghiệm chính
#     (2 Model: GPT-OSS 120B & Gemini 3.8 Flash × 3 Phương pháp: No-RAG / RAG / CoVe, 50 linh kiện)
python -m src.experiments.runner --exp exp2 --profile adapted

# 2.3 (Tùy chọn - Phát triển offline) Chạy thử bằng Ollama Qwen3 8B trên 3 linh kiện đầu tiên:
python -m src.experiments.runner --exp exp2 --dev --limit 3
```

### Bước 3 – Đánh giá (Human Evaluation & LLM-as-a-Judge) và Xuất Báo cáo
```powershell
# 3.1 Mở giao diện Web (Streamlit) để Người gán nhãn mù (Primary Evaluation)
#     - Exp I : Ưu tiên 4 người gán nhãn (Coverage@1/@3, Precision@1/@3)
#     - Exp II: 3 người gán nhãn (Coverage@2, Precision@2, Factual Error, Rating 1-5, Kendall's Tau, Krippendorff's Alpha)
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py

# 3.2 Chạy Giám khảo LLM độc lập (qwen/qwen3.8-27b - Supplementary Evaluation, lưu tách biệt kind="llm")
python -m src.evaluation.llm_judge --exp exp1
python -m src.evaluation.llm_judge --exp exp2

# 3.3 Tổng hợp báo cáo & vẽ biểu đồ (tách riêng Human Evaluation, LLM-as-Judge và Human-LLM Agreement)
#     Kết quả xuất ra: results/metrics/report.md, results/metrics/metrics.json, results/plots/*.png
python -m src.evaluation.report

# 3.4 Xuất toàn bộ dữ liệu từ SQLite (database/experiments.db) ra file JSONL/CSV
python -m src.storage.export
```

---

## 6. Xử lý Lỗi Thường gặp (Troubleshooting)

| Lỗi gặp phải | Nguyên nhân | Cách khắc phục |
|---|---|---|
| `ModuleNotFoundError: No module named 'yaml'` (hoặc `faiss`, `bm25s`, `pydantic`...) | Chưa kích hoạt môi trường ảo `.venv` nên PowerShell gọi nhầm Python hệ thống | Chạy `.\.venv\Scripts\Activate.ps1` trước, hoặc dùng `.\.venv\Scripts\python.exe -m ...` |
| `... cannot be loaded because running scripts is disabled on this system` | PowerShell chặn chạy file `.ps1` mặc định trên Windows | Chạy `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` rồi kích hoạt lại `.venv` |
| `GROQ_API_KEY` / `GEMINI_API_KEY` missing | Chưa tạo file `.env` hoặc chưa điền API key | Mở file `.env` ở thư mục gốc và điền key hợp lệ, sau đó test bằng `python -m src.llm.check --ping` |
| Lỗi thiếu file chỉ mục khi chạy `exp2` | Chưa chạy bước tạo passage hoặc lập chỉ mục `indexing` | Chạy lần lượt các lệnh ở **Bước 1** (hoặc dùng cờ `--mock-rag` nếu chỉ muốn test luồng code) |

---

## 7. Bảng Ánh xạ Lệnh CLI (Cũ → Mới)

| Tác vụ | Lệnh cũ (đã ngừng dùng) | Lệnh mới |
|---|---|---|
| Tải Wikipedia dump | `python -m src.wiki_download` | `python -m src.rag.ingestion download` |
| Lọc bài viết Wikipedia | `python -m src.wiki_extract` | `python -m src.rag.ingestion extract` |
| Cắt passage 200/50 từ | `python -m src.build_passages` | `python -m src.rag.chunking` |
| Lập chỉ mục BM25 + Dense | `python -m src.build_index` | `python -m src.rag.indexing --all [--profile adapted\|reproduction]` |
| Truy vấn thử Hybrid RAG | `python -m src.retriever brake disc` | `python -m src.rag.hybrid_retriever brake disc` |
| Kiểm tra chất lượng retrieval | `python -m src.retrieval_check` | `python -m src.evaluation.retrieval_check` |
| Chạy Experiment I | `python -m src.run_exp1` | `python -m src.experiments.runner --exp exp1` |
| Chạy Experiment II | `python -m src.run_exp2` | `python -m src.experiments.runner --exp exp2` |
| Chạy LLM-as-a-Judge | `python -m src.judge --exp exp2` | `python -m src.evaluation.llm_judge --exp exp2` |
| Tổng hợp báo cáo & biểu đồ | `python -m src.evaluate` | `python -m src.evaluation.report` |
| Giao diện gán nhãn / UI | `streamlit run src/annotate_app.py` | `streamlit run app/streamlit_app.py` |
| Kiểm thử toàn tuyến | `python tests/smoke_test.py` | `python -m pytest` |
