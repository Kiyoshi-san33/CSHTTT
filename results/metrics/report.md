# Báo cáo thực nghiệm dự đoán vật liệu linh kiện ô tô

## 1. Kho ngữ liệu Wikipedia

|                |   Paper (Wagner et al., 2026) |   Thực nghiệm |
|:---------------|------------------------------:|--------------:|
| Số bài viết    |                         32368 |         24205 |
| Tiêu chí (1)   |                         29425 |         16802 |
| Tiêu chí (2)   |                          2943 |          7403 |
| Độ dài TB (từ) |                          4233 |          3007 |
| Số passage     |                        458972 |        489318 |


## 2. Experiment I – Adapted Baseline (GPT-OSS 120B, Few-shot)

> **Lưu ý phương pháp:** Sử dụng nguyên văn prompt Few-shot ở Phụ lục A nhưng chạy trên `openai/gpt-oss-120b` (Adapted Baseline), không phải bản tái lập trực tiếp `Mixtral-8x22B`.

### 2.1 Đánh giá chính thức bởi Người (Human Evaluation)

_Chưa có nhãn cho nhóm này._

### 2.2 Đánh giá bổ sung bởi Giám khảo LLM (LLM-as-Judge) (100 linh kiện, 1 annotator: judge_qwen38_27b)

#### Bảng 1 – So sánh với nghiên cứu trước (Adapted Baseline: GPT-OSS 120B, Few-shot Appendix A)

|                                                                                         | COVERAGE@1   | COVERAGE@3   | PRECISION@1   | PRECISION@3   |
|:----------------------------------------------------------------------------------------|:-------------|:-------------|:--------------|:--------------|
| Schlipf (2022)                                                                          | 73%          | 40%          | 45%           | 14%           |
| Eichel et al. (2023)                                                                    | 93%          | 73%          | 62%           | 28%           |
| Paper (Mixtral-8x22B)                                                                   | 100%         | 100%         | 97%           | 71%           |
| Adapted Baseline – GPT-OSS 120B (2.2 Đánh giá bổ sung bởi Giám khảo LLM (LLM-as-Judge)) | 100.0%       | —            | 90.0%         | —             |


#### Coverage@n / Precision@n theo ngưỡng số annotator đồng ý

|           | @1     |
|:----------|:-------|
| COVERAGE  | 100.0% |
| PRECISION | 90.0%  |


#### IAA (Eq. 2) = **nan** (paper: 0.69)

_Cần >= 2 annotator để tính ma trận IAA._


## 3. Experiment II – Dự đoán kèm giải thích (2 LLM × No-RAG / RAG / CoVe)

> **Quy tắc tách nguồn nhãn:** Các chỉ số chính thức (Coverage@2, Precision@2, Factual Error, Human Rating 1–5, Kendall's τ, Krippendorff's α) chỉ tính từ phiếu của **Người**. Kết quả của **LLM-as-Judge** (`qwen/qwen3.8-27b`) được trình bày ở mục riêng.

### 3.1 Đánh giá chính thức bởi Người (Primary Human Evaluation)

_Chưa có nhãn cho nhóm này._

### 3.2 Đánh giá bổ sung bởi Giám khảo LLM (Supplementary LLM-as-Judge) (300 output, 1 annotator: judge_qwen38_27b; ngưỡng đồng ý = @1) _(hiện có 1 annotator nên tính @1; cần >= 2 để tính @2)_

#### Bảng 2a – Task 1: COVERAGE@1

|              | gemini_flash   | gpt_oss_120b   | Average   |
|:-------------|:---------------|:---------------|:----------|
| norag        | 100.0%         | 100.0%         | 100.0%    |
| rag@adapted  | 100.0%         | 100.0%         | 100.0%    |
| cove@adapted | 100.0%         | 100.0%         | 100.0%    |


#### Bảng 2b – Task 1: PRECISION@1

|              | gemini_flash   | gpt_oss_120b   | Average   |
|:-------------|:---------------|:---------------|:----------|
| norag        | 99.6%          | 96.3%          | 98.0%     |
| rag@adapted  | 98.4%          | 97.0%          | 97.5%     |
| cove@adapted | 99.0%          | 96.7%          | 97.7%     |


#### Bảng 3a – Task 2: Tỉ lệ lỗi sự thật / Hallucination Rate (trung bình theo annotator)

|              | gemini_flash   | gpt_oss_120b   | Average   |
|:-------------|:---------------|:---------------|:----------|
| norag        | 4.0%           | 22.0%          | 13.0%     |
| rag@adapted  | 6.0%           | 24.0%          | 15.0%     |
| cove@adapted | 10.0%          | 22.0%          | 16.0%     |


_Theo tiêu chí 'ít nhất 1 annotator đánh dấu lỗi':_

|              | gemini_flash   | gpt_oss_120b   | Average   |
|:-------------|:---------------|:---------------|:----------|
| norag        | 4.0%           | 22.0%          | 13.0%     |
| rag@adapted  | 6.0%           | 24.0%          | 15.0%     |
| cove@adapted | 10.0%          | 22.0%          | 16.0%     |


#### Bảng 4a – Task 3: Điểm chất lượng dự đoán vật liệu (1–5)

|              |   gemini_flash |   gpt_oss_120b |   Average |
|:-------------|---------------:|---------------:|----------:|
| norag        |           4.74 |           4.64 |      4.69 |
| rag@adapted  |           4.16 |           4.76 |      4.46 |
| cove@adapted |           4.38 |           4.6  |      4.49 |


#### Bảng 4b – Task 4: Điểm chất lượng lời giải thích (1–5)

|              |   gemini_flash |   gpt_oss_120b |   Average |
|:-------------|---------------:|---------------:|----------:|
| norag        |           4.88 |           4.66 |      4.77 |
| rag@adapted  |           4.64 |           4.6  |      4.62 |
| cove@adapted |           4.72 |           4.56 |      4.64 |


#### Bảng 3b – Task 5: Vị trí xếp hạng trung bình (0 = tốt nhất, thấp hơn = tốt hơn)

|              |   gemini_flash |   gpt_oss_120b |   Average |
|:-------------|---------------:|---------------:|----------:|
| norag        |           2.18 |           2.98 |      2.58 |
| rag@adapted  |           3.52 |           1.68 |      2.6  |
| cove@adapted |           3.26 |           1.38 |      2.32 |


#### Bảng 7 – Coverage@n & Precision@n theo các ngưỡng n

|              | COV@1   | PREC@1   |
|:-------------|:--------|:---------|
| norag        | 100.0%  | 98.0%    |
| rag@adapted  | 100.0%  | 97.5%    |
| cove@adapted | 100.0%  | 97.7%    |


#### Bảng 9 – PRECISION@1 của vật liệu đầu vs cuối danh sách

|              | First material   | Last material   |
|:-------------|:-----------------|:----------------|
| norag        | 100.0%           | 94.9%           |
| rag@adapted  | 99.0%            | 94.9%           |
| cove@adapted | 99.0%            | 96.0%           |


#### Bảng 6 – Độ đồng thuận giữa các annotator trong nhóm

| Chỉ số                  |   Thực nghiệm |   Paper |
|:------------------------|--------------:|--------:|
| IAA Eq.2 (Task 1)       |           nan |    0.63 |
| Krippendorff α (Task 3) |           nan |   -0.22 |
| Krippendorff α (Task 4) |           nan |   -0.23 |
| Kendall τ (Task 5)      |           nan |    0.24 |


#### Kiểm định Wilcoxon (No-RAG so với RAG / CoVe)

| so sánh               | chỉ số             |   n cặp |   TB norag |   TB phương pháp |   p-value |
|:----------------------|:-------------------|--------:|-----------:|-----------------:|----------:|
| norag vs cove@adapted | material_rating    |     100 |       4.69 |             4.49 |    0.0117 |
| norag vs cove@adapted | explanation_rating |     100 |       4.77 |             4.64 |    0.1044 |
| norag vs rag@adapted  | material_rating    |     100 |       4.69 |             4.46 |    0.0071 |
| norag vs rag@adapted  | explanation_rating |     100 |       4.77 |             4.62 |    0.0808 |

