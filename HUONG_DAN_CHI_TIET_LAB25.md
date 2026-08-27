# HƯỚNG DẪN CHI TIẾT TOÀN DIỆN: LAB 25 — GPU FINOPS OPTIMIZATION

> **Khóa học:** AICB · Phase 2 · Track 2 (Infrastructure) · Day 25  
> **Dự án:** Tối ưu hóa chi phí GPU (GPU FinOps Workshop)  
> **Mục tiêu:** Cắt giảm chi phí GPU từ **40% đến 95%** đo bằng đơn vị chuẩn `$/1M-token`.  
> **Môi trường:** Chạy 100% trên Python cục bộ (không cần GPU vật lý, không cần tài khoản Cloud, không cần API Key).

---

## MỤC LỤC

1. [Ý NGHĨA & BỐI CẢNH BÀI LAB](#1-ý-nghĩa--bối-cảnh-bài-lab)
   - [1.1. Bối cảnh thực tế tại doanh nghiệp AI (NimbusAI)](#11-bối-cảnh-thực-tế-tại-doanh-nghiệp-ai-nimbusai)
   - [1.2. Chuyển dịch tư duy: $/GPU-giờ sang $/1M-token](#12-chuyển-dịch-tư-duy-gpu-giờ-sang-1m-token)
   - [1.3. Cạm bẫy "GPU-Util Lie" & Mô hình Roofline](#13-cạm-bẫy-gpu-util-lie--mô-hình-roofline)
   - [1.4. Bốn đòn bẩy tối ưu hóa chi phí cốt lõi](#14-bốn-đòn-bẩy-tối-ưu-hóa-chi-phí-cốt-lõi)
   - [1.5. Quản trị phân bổ chi phí & Tiêu chuẩn FOCUS](#15-quản-trị-phân-bổ-chi-phí--tiêu-chuẩn-focus)
   - [1.6. Tính bền vững & Kinh tế học Năng lượng (Green FinOps)](#16-tính-bền-vững--kinh-tế-học-năng-lượng-green-finops)
2. [CẤU TRÚC DỰ ÁN & GIẢI THÍCH DỮ LIỆU ĐẦU VÀO](#2-cấu-trúc-dự-án--giải-thích-dữ-liệu-đầu-vào)
   - [2.1. Cây thư mục dự án](#21-cây-thư-mục-dự-án)
   - [2.2. Chi tiết 4 tập dữ liệu trong thư mục `data/`](#22-chi-tiết-4-tập-dữ-liệu-trong-thư-mục-data)
   - [2.3. Vai trò của các module trong gói `finops/`](#23-vai-trò-của-các-module-trong-gói-finops)
3. [HƯỚNG DẪN CÀI ĐẶT MÔI TRƯỜNG](#3-hướng-dẫn-cài-đặt-môi-trường)
4. [HƯỚNG DẪN THỰC HIỆN TỪNG BƯỚC (MISSIONS 1 – 5)](#4-hướng-dẫn-thực-hiện-từng-bước-missions-1--5)
   - [Bước 1: Mission 1 — Kiểm toán hiệu quả GPU (Efficiency Audit)](#bước-1-mission-1--kiểm-toán-hiệu-quả-gpu-efficiency-audit)
   - [Bước 2: Mission 2 — Đòn bẩy chi phí Inference (Inference Levers)](#bước-2-mission-2--đòn-bẩy-chi-phí-inference-inference-levers)
   - [Bước 3: Mission 3 — Chiến lược mua sắm GPU (Purchasing Strategy)](#bước-3-mission-3--chiến-lược-mua-sắm-gpu-purchasing-strategy)
   - [Bước 4: Mission 4 — Phân bổ chi phí & Tiêu chuẩn FOCUS (Cost Allocation)](#bước-4-mission-4--phân-bổ-chi-phí--tiêu-chuẩn-focus-cost-allocation)
   - [Bước 5: Mission 5 — Báo cáo tổng hợp & Biểu đồ Waterfall (Optimization Report)](#bước-5-mission-5--báo-cáo-tổng-hợp--biểu-đồ-waterfall-optimization-report)
5. [HƯỚNG DẪN THỰC HIỆN PHẦN MỞ RỘNG "YOUR TURN" (20 ĐIỂM)](#5-hướng-dẫn-thực-hiện-phần-mở-rộng-your-turn-20-điểm)
   - [Extension 1: Tinh chỉnh chính sách `recommend_tier()`](#extension-1-tinh-chỉnh-chính-sách-recommend_tier)
   - [Extension 2: Right-sizing theo MBU cho Memory-bound Workloads](#extension-2-right-sizing-theo-mbu-cho-memory-bound-workloads)
   - [Extension 3: Điểm hòa vốn của Prompt Caching (`cache_is_worth_it`)](#extension-3-điểm-hòa-vốn-của-prompt-caching-cache_is_worth_it)
   - [Extension 4: Quản trị & Phân bổ ngân sách Reasoning Traffic](#extension-4-quản-trị--phân-bổ-ngân-sách-reasoning-traffic)
   - [Extension 5: Lập lịch huấn luyện nhận thức Carbon (Carbon-aware Scheduling)](#extension-5-lập-lịch-huấn-luyện-nhận-thức-carbon-carbon-aware-scheduling)
6. [HƯỚNG DẪN KIỂM THỬ, XÁC MINH & ĐÁNH GIÁ (RUBRIC)](#6-hướng-dẫn-kiểm-thử-xác-minh--đánh-giá-rubric)
   - [6.1. Xác minh tự động qua `verify.py` (30 điểm)](#61-xác-minh-tự-động-qua-verifypy-30-điểm)
   - [6.2. Chạy bộ Unit Tests với `pytest` (20 điểm)](#62-chạy-bộ-unit-tests-với-pytest-20-điểm)
   - [6.3. Tiêu chí chấm Báo cáo kỹ thuật (30 điểm) & Phần mở rộng (20 điểm)](#63-tiêu-chí-chấm-báo-cáo-kỹ-thuật-30-điểm--phần-mở-rộng-20-điểm)
7. [HƯỚNG DẪN VIẾT BÁO CÁO WRITE-UP & CHUẨN BỊ NỘP BÀI](#7-hướng-dẫn-viết-báo-cáo-write-up--chuẩn-bị-nộp-bài)
8. [CÁC LỖI THƯỜNG GẶP & CÁCH XỬ LÝ (TROUBLESHOOTING)](#8-các-lỗi-thường-gặp--cách-xử-lý-troubleshooting)

---

## 1. Ý NGHĨA & BỐI CẢNH BÀI LAB

### 1.1. Bối cảnh thực tế tại doanh nghiệp AI (NimbusAI)
Trong kỷ nguyên Generative AI, chi phí GPU đám mây (Cloud GPU) là gánh nặng tài chính lớn nhất đối với các startup và doanh nghiệp AI. 
Bạn đóng vai trò là một **FinOps Engineer** tại *NimbusAI*. Doanh nghiệp đang đối mặt với việc hóa đơn GPU tăng phi mã hàng tháng mà không rõ nguyên nhân, đe dọa sự sống còn của dự án. 

Mục tiêu cốt lõi của bài lab là trang bị cho bạn năng lực:
1. **Phát hiện lãng phí ngầm** trong hạ tầng GPU (GPU để không, GPU chạy không hiệu quả).
2. **Áp dụng các đòn bẩy kỹ thuật & tài chính** để cắt giảm từ **40% đến 95%** chi phí.
3. **Thiết lập cơ chế quản trị chi phí minh bạch** (Showback/Chargeback) theo chuẩn công nghiệp **FOCUS** của FinOps Foundation.
4. **Tích hợp yếu tố môi trường (Green AI)** vào quyết định triển khai hạ tầng.

---

### 1.2. Chuyển dịch tư duy: $/GPU-giờ sang $/1M-token

Một sai lầm phổ biến của các kỹ sư hạ tầng truyền thống là tối ưu hóa theo chỉ số `$/GPU-giờ` (đơn giá thuê GPU trên đám mây). Tuy nhiên, trong lĩnh vực LLM/GenAI:
- **`$/GPU-giờ` (Tư duy Hạ tầng):** Chỉ cho biết bạn trả bao nhiêu tiền để giữ GPU hoạt động trong 1 giờ. Nếu GPU hoạt động không tối ưu hoặc bị nghẽn (bottleneck), bạn vẫn phải trả đủ tiền dù khối lượng tính toán thực tế tạo ra rất ít.
- **`$/1M-token` (Tư duy Đơn vị Kinh tế - Unit Economics):** Cho biết doanh nghiệp phải trả bao nhiêu tiền để phục vụ được **1 triệu token** (đơn vị giá trị nghiệp vụ thực tế mang lại cho người dùng).

> **Ý nghĩa:** Một GPU giá $2.5/giờ nếu phục vụ được 100k token/giờ sẽ có chi phí $25/1M-token. Trong khi một cụm GPU giá $4.0/giờ nếu được tối ưu tốt và phục vụ 1M token/giờ sẽ chỉ tốn $4/1M-token (rẻ hơn hơn 6 lần về mặt kinh tế!).

---

### 1.3. Cạm bẫy "GPU-Util Lie" & Mô hình Roofline

#### Hiện tượng "GPU-Util Lie"
Khi chạy lệnh `nvidia-smi`, chỉ số `GPU-Util %` (GPU Utilization) thực chất chỉ đo **tỷ lệ phần trăm thời gian mà nhân đồ họa (clock) có hoạt động**, hoàn toàn **KHÔNG** phản ánh hiệu quả tính toán thực tế.
- Một GPU có thể báo `GPU-Util: 98%` nhưng thực chất các nhân Tensor Core đang phải dừng chờ nạp dữ liệu từ bộ nhớ (Memory Stall) hoặc bị nghẽn I/O.
- Trong bài lab, `gpu-h100-4` có `GPU-Util = 98%` nhưng **MFU chỉ đạt ~0.20 (20%)**. Doanh nghiệp đang trả 100% tiền thuê H100 nhưng chỉ nhận lại 1/5 hiệu năng lý thuyết!

#### Các thước đo chuẩn xác:
- **MFU (Model FLOPs Utilization):** Tỷ lệ giữa lượng tính toán thực tế đạt được (`achieved_tflops`) so với năng lực tính toán đỉnh lý thuyết (`peak_tflops_fp16`). 
  $$\text{MFU} = \frac{\text{achieved\_tflops}}{\text{peak\_tflops}}$$
  *(Ngưỡng tốt trong huấn luyện LLM: 35% - 45%, trên 50% là xuất sắc).*
- **MBU (Model Bandwidth Utilization):** Tỷ lệ giữa băng thông bộ nhớ HBM thực tế đạt được so với băng thông đỉnh (`peak_bw_tbs`). Thước đo này dùng cho các tác vụ nghẽn bộ nhớ.
- **Mô hình Roofline & Phân tách Prefill/Decode:**
  - Điểm gờ (Ridge Point) của NVIDIA H100 là $\approx 295 \text{ FLOP/byte}$.
  - Giai đoạn **Prefill** (xử lý prompt đầu vào): Cường độ tính toán cao ($\approx 455 \text{ FLOP/byte}$) $\rightarrow$ **Compute-bound** (Nghẽn tính toán, cần tối ưu MFU).
  - Giai đoạn **Decode** (sinh từng token tiếp theo): Cường độ tính toán rất thấp ($\approx 1 - 2 \text{ FLOP/byte}$) $\rightarrow$ **Memory-bound** (Nghẽn băng thông bộ nhớ, cần tối ưu MBU).

---

### 1.4. Bốn đòn bẩy tối ưu hóa chi phí cốt lõi

1. **Inference Levers (Đòn bẩy Tầng Suy luận):**
   - **Model Cascading (Phân tầng Model):** Định tuyến ~80% các yêu cầu đơn giản sang model nhỏ giá rẻ ($0.20/$0.40 trên 1M token), chỉ chuyển các câu hỏi phức tạp sang model lớn ($3.00/$15.00 trên 1M token).
   - **Prompt Caching:** Lưu cache các đoạn prompt dài dùng chung (System Prompt, Context tài liệu RAG). Token đọc từ cache được giảm giá tới **90%** (chỉ tính 10% đơn giá input).
   - **Batch API:** Gom các tác vụ phi thời gian thực (như chạy đánh giá Offline Eval, xử lý tài liệu nền) để chạy theo lô, được nhà cung cấp giảm giá **50%**.
   - **Discount Stacking:** Khi kết hợp Batch API + 100% Cache Hit:
     $$\text{Chi phí hiệu dụng} = 0.50 \times 0.10 = 0.05 \quad (\text{Giảm tới } 95\%!)$$

2. **Purchasing Strategy (Chiến lược Mua sắm Hạ tầng):**
   - **On-Demand:** Mua theo giờ, linh hoạt nhưng đắt nhất.
   - **Spot Instance:** Giá rẻ hơn 40–60%, nhưng có rủi ro bị thu hồi (preemption). Thích hợp cho các Job huấn luyện có cơ chế lưu Checkpoint tự động.
   - **Reserved Instance (Cam kết dài hạn 1–3 năm):** Giảm giá ~45%.
   - **Điểm hòa vốn (Break-even Utilization):**
     $$\text{Break-even} = 1 - \text{Discount} = 1 - 0.45 = 55\% \quad (\approx 13.2 \text{ giờ/ngày})$$
     Nếu GPU chạy $\ge 13.2$ giờ/ngày đều đặn thì nên mua Reserved; nếu chạy ít hơn thì On-demand rẻ hơn.

3. **Right-sizing (Chọn đúng kích cỡ):**
   - Hạ cấp các GPU bị "GPU-Util Lie" xuống các dòng GPU phù hợp hơn (ví dụ: chuyển từ H100 sang A100/A10G khi tác vụ không khai thác được Tensor Core của H100).

4. **Loại bỏ lãng phí Idle (Idle Waste Elimination):**
   - Tự động tắt hoặc thu hồi các GPU nhàn rỗi qua đêm (utilization < 10%).

---

### 1.5. Quản trị phân bổ chi phí & Tiêu chuẩn FOCUS

- **Thang trưởng thành FinOps (Maturity Ladder):**
  $$\text{Visibility (Thấy dữ liệu)} \longrightarrow \text{Showback (Báo cáo minh bạch)} \longrightarrow \text{Chargeback (Thu phí thực tế)}$$
- **Nguyên tắc "Cổng Chargeback" (Chargeback Gate):**
  Chỉ được phép thực hiện thu phí nội bộ (Chargeback) khi **Tag Coverage $\ge 80\%$**. Nếu tỷ lệ gắn tag thấp hơn, việc phân bổ chi phí sẽ không chính xác và gây tranh cãi giữa các phòng ban.
- **Tiêu chuẩn FOCUS (FinOps Open Cost & Usage Specification):**
  Chuẩn hóa các cột dữ liệu chi phí (`BillingAccountId`, `ChargePeriodStart`, `ServiceCategory`, `BilledCost`, tags...) giúp doanh nghiệp quản lý chi phí đa nền tảng nhất quán.

---

### 1.6. Tính bền vững & Kinh tế học Năng lượng (Green FinOps)

- Các mô hình suy luận sâu (Reasoning models như o1/r1) tiêu thụ năng lượng gấp **$\approx 80$ lần** so với truy vấn thông thường.
- Cường độ phát thải Carbon (`gCO2/kWh`) và giá điện (`$/kWh`) khác nhau rất lớn giữa các vùng địa lý:
  - Vùng **europe-north1** (Na Uy): Sử dụng thủy điện $\rightarrow$ Carbon cực thấp ($30 \text{ gCO2/kWh}$), giá rẻ ($0.09/kWh).
  - Vùng **europe-central2** (Ba Lan): Nhiệt điện than $\rightarrow$ Phát thải cực cao ($660 \text{ gCO2/kWh}$), giá đắt ($0.18/kWh).
- **Ý nghĩa:** Chuyển dịch vùng đặt máy chủ cho các batch job vừa giúp cắt giảm chi phí vừa giảm phát thải carbon hàng tấn mỗi năm.

---

## 2. CẤU TRÚC DỰ ÁN & GIẢI THÍCH DỮ LIỆU ĐẦU VÀO

### 2.1. Cây thư mục dự án

```
Day25-Track2-GPU-FinOps-Lab/
├── data/                           # Dữ liệu đầu vào tổng hợp
│   ├── generate.py                 # Script sinh dữ liệu tất định (seed=25)
│   ├── price_catalog.csv           # Bảng giá 7 dòng GPU thị trường
│   ├── gpu_telemetry.csv           # Dữ liệu đo đạc 11 GPU trong 24 giờ
│   ├── token_usage.csv             # Lịch sử 2,400 lượt gọi suy luận LLM
│   └── workloads.csv               # Danh sách 8 tác vụ AI của công ty
├── finops/                         # Thư viện lõi tính toán FinOps
│   ├── metrics.py                  # Công thức MFU, MBU, roofline, lọc GPU-Util lie
│   ├── pricing.py                  # Tính tiền request, $/1M-token, discount stack, tier
│   ├── allocation.py               # Phân bổ tag, tính tag coverage, xuất chuẩn FOCUS
│   ├── sustainability.py           # Tính năng lượng (Wh), carbon phát thải, chọn vùng
│   └── report.py                   # Render Markdown report & vẽ biểu đồ Waterfall
├── missions/                       # 5 kịch bản thực thi chính
│   ├── _common.py                  # Helper đọc file CSV & chuyển đổi kiểu
│   ├── m1_efficiency_audit.py      # Mission 1: Kiểm toán hiệu quả GPU
│   ├── m2_inference_levers.py      # Mission 2: Đòn bẩy chi phí suy luận
│   ├── m3_purchasing.py            # Mission 3: Chiến lược mua sắm GPU
│   ├── m4_allocation.py            # Mission 4: Phân bổ chi phí theo phòng ban
│   ├── m5_report.py                # Mission 5: Báo cáo tổng hợp Baseline vs Optimized
│   └── run_all.py                  # Chạy toàn bộ M1 -> M5 liên tiếp
├── tests/                          # 15 bài unit & integration tests (pytest)
│   ├── test_metrics.py             # Test công thức MFU, MBU, lies, idle
│   ├── test_pricing.py             # Test tính giá, discount, break-even
│   ├── test_allocation.py          # Test tag coverage, FOCUS
│   ├── test_report.py              # Test sinh markdown report
│   └── test_data_and_missions.py   # Test tích hợp toàn bộ pipeline
├── outputs/                        # Thư mục chứa kết quả xuất ra
│   ├── report.md                   # Báo cáo tối ưu hóa chi phí
│   ├── savings.png                 # Biểu đồ cột tiết kiệm chi phí
│   └── focus_export.csv            # File xuất chi phí chuẩn FOCUS
├── bonus/                          # Phần mở rộng tùy chọn
│   ├── litellm_tracker/            # Proxy theo dõi budget cap
│   ├── local_model/                # Benchmark model cục bộ trên CPU
│   └── docker/                     # Dashboard Grafana + Prometheus
├── verify.py                       # Script kiểm tra tự động 11/11 tiêu chí
├── requirements.txt                # Thư viện phụ thuộc (pandas, matplotlib, pytest)
├── README.md                       # Tài liệu tổng quan lab
├── Guide.md                        # Hướng dẫn gốc của lab
└── Rubric.md                       # Thang điểm chi tiết
```

---

### 2.2. Chi tiết 4 tập dữ liệu trong thư mục `data/`

1. **`price_catalog.csv` (7 dòng GPU):**
   - Các cột: `gpu_type`, `provider_class`, `on_demand_hr`, `spot_hr`, `reserved_1yr_hr`, `reserved_3yr_hr`, `hbm_gb`, `peak_tflops_fp16`, `peak_tflops_fp4`, `peak_bw_tbs`, `watts`.
   - Chứa thông số kỹ thuật đỉnh và các mức giá thuê theo giờ của các GPU phổ biến: H100, H200, A100, A10G, L4, B200, MI300X.
2. **`gpu_telemetry.csv` (11 GPU $\times$ 24 giờ = 264 bản ghi):**
   - Các cột: `ts`, `gpu_id`, `gpu_type`, `gpu_util_pct`, `sm_active_pct`, `tensor_active_pct`, `dram_active_pct`, `power_w`, `mem_used_gb`, `achieved_tflops`, `achieved_bw_tbs`, `workload`.
   - Ghi lại nhật ký hoạt động thực tế từng giờ của đội ngũ GPU.
3. **`token_usage.csv` (2,400 requests):**
   - Các cột: `ts`, `model`, `team`, `project`, `route_tier`, `input_tokens`, `output_tokens`, `cached_input_tokens`, `is_batch`, `is_reasoning`, `latency_ms`.
   - Dữ liệu gọi API của các team: `search`, `rag`, `assistant`, `eval`.
4. **`workloads.csv` (8 jobs):**
   - Các cột: `job_id`, `team`, `kind`, `hours_per_day`, `days`, `interruptible`, `gpu_type`, `num_gpus`.
   - Các tác vụ huấn luyện/suy luận định kỳ của công ty.

---

### 2.3. Vai trò của các module trong gói `finops/`

| Module | Hàm chính | Vai trò kỹ thuật |
|---|---|---|
| `metrics.py` | `compute_mfu`, `compute_mbu`, `roofline_regime`, `flag_util_lies`, `idle_waste_usd` | Đo đạc hiệu năng tính toán thực chất và phát hiện lãng phí ngầm |
| `pricing.py` | `request_cost`, `dollars_per_million`, `discount_stack`, `break_even_utilization`, `recommend_tier`, `spot_checkpoint_cost` | Mô hình hóa kinh tế học token, chiết khấu và tối ưu chiến lược mua sắm |
| `allocation.py` | `cost_by_tag`, `tag_coverage`, `chargeback_ready`, `to_focus_rows` | Quản trị chi phí, kiểm tra độ phủ tag và chuẩn hóa dữ liệu |
| `sustainability.py` | `wh_per_query`, `carbon_g`, `energy_cost_usd`, `tokens_per_watt` | Đo lường năng lượng, phát thải carbon và tối ưu hóa vị trí địa lý |
| `report.py` | `build_report`, `savings_waterfall` | Tự động hóa tạo báo cáo Markdown và xuất biểu đồ trực quan |

---

## 3. HƯỚNG DẪN CÀI ĐẶT MÔI TRƯỜNG

Mở terminal tại thư mục gốc của dự án và thực hiện các lệnh sau:

### Trên Windows (PowerShell):
```powershell
# 1. Tạo môi trường ảo Python
python -m venv .venv

# 2. Kích hoạt môi trường ảo
.venv\Scripts\Activate.ps1

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### Trên macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 4. HƯỚNG DẪN THỰC HIỆN TỪNG BƯỚC (MISSIONS 1 – 5)

### Khởi tạo dữ liệu gốc
Trước khi chạy các mission, hãy đảm bảo dữ liệu tổng hợp đã được sinh đầy đủ:
```bash
python data/generate.py
```

---

### Bước 1: Mission 1 — Kiểm toán hiệu quả GPU (Efficiency Audit)

- **Mục tiêu:** Tính toán chỉ số MFU/MBU cho từng GPU, phát hiện GPU "nói dối" (`GPU-Util $\ge 90\%$` nhưng `MFU < 30%`), và tính lượng tiền lãng phí khi để GPU chạy không tải qua đêm.
- **Thực thi lệnh:**
  ```bash
  python missions/m1_efficiency_audit.py
  ```
- **Nguyên lý hoạt động trong code (`missions/m1_efficiency_audit.py`):**
  1. Đọc dữ liệu từ `gpu_telemetry.csv` và `price_catalog.csv`.
  2. Với mỗi GPU, tính `achieved_tflops / peak_tflops_fp16` để ra MFU.
  3. Đếm số giờ GPU chạy không tải (`gpu_util_pct < 10%`).
  4. Dùng `metrics.flag_util_lies()` để tìm ra GPU có `gpu_util_pct >= 90%` nhưng `MFU < 0.30`.
- **Kết quả thu được:**
  - Phát hiện `gpu-h100-4` bị "nói dối" (GPU-Util 98.0%, MFU chỉ 0.202) và `gpu-a10g-1` (Util 97.0%, MFU 0.270).
  - Lãng phí do GPU nhàn rỗi (idle waste) là **$20.00/ngày** $\rightarrow$ **$600/tháng**.

---

### Bước 2: Mission 2 — Đòn bẩy chi phí Inference (Inference Levers)

- **Mục tiêu:** So sánh chi phí giữa triển khai ngây thơ (Baseline - toàn bộ đẩy vào model lớn, không cache, không batch) và triển khai tối ưu (Optimized - phân tầng Cascade + Prompt Caching 90% + Batch API 50%).
- **Thực thi lệnh:**
  ```bash
  python missions/m2_inference_levers.py
  ```
- **Nguyên lý tính toán trong code (`finops/pricing.py`):**
  - **Baseline:**
    $$\text{Cost} = \frac{\text{Input}}{10^6} \times \$3.00 + \frac{\text{Output}}{10^6} \times \$15.00$$
  - **Optimized:**
    - Tuyến đường nhỏ: Input $0.20/1M, Output $0.40/1M.
    - Input được chia thành: `uncached_input` (tính 100% giá) + `cached_input` (chỉ tính 10% giá).
    - Nếu `is_batch == True`: Toàn bộ hóa đơn được nhân thêm $0.50$ (-50%).
- **Kết quả thu được:**
  - Baseline: **$45.23/ngày** ($\approx \$6.49/1\text{M-token}$).
  - Optimized: **$7.86/ngày** ($\approx \$1.13/1\text{M-token}$).
  - Tiết kiệm được: **82.6%** chi phí suy luận.

---

### Bước 3: Mission 3 — Chiến lược mua sắm GPU (Purchasing Strategy)

- **Mục tiêu:** Chọn đúng hình thức thuê GPU (On-Demand, Spot kèm Checkpoint, hoặc Reserved 3 năm) cho 8 workload của doanh nghiệp.
- **Thực thi lệnh:**
  ```bash
  python missions/m3_purchasing.py
  ```
- **Quy tắc phân loại (`pricing.recommend_tier`):**
  1. Nếu công việc có thể gián đoạn (`interruptible == True`) và không chạy 24/7 $\rightarrow$ Chọn **`spot`**.
  2. Nếu thời gian chạy $\ge 13.2 \text{ giờ/ngày}$ (Duty Cycle $\ge 55\%$ - Điểm hòa vốn Reserved) $\rightarrow$ Chọn **`reserved`**.
  3. Còn lại $\rightarrow$ Chọn **`on_demand`**.
- **Mô phỏng Spot Checkpoint (`pricing.spot_checkpoint_cost`):**
  - Tính toán thêm 3% overhead ghi checkpoint và thời gian tính toán lại (rework) khi bị thu hồi (với tỷ lệ rủi ro 5%/giờ). Spot vẫn tiết kiệm vượt trội cho các job training.
- **Kết quả thu được:**
  - Chi phí On-Demand: **$45,000/tháng**.
  - Chi phí sau tối ưu: **$27,422/tháng**.
  - Tiết kiệm được: **39.1%** chi phí mua sắm hạ tầng.

---

### Bước 4: Mission 4 — Phân bổ chi phí & Tiêu chuẩn FOCUS (Cost Allocation)

- **Mục tiêu:** Nhóm chi phí theo `team` và `project`, kiểm tra tỷ lệ gắn tag (Tag Coverage), mở cổng Chargeback và xuất dữ liệu ra file CSV chuẩn quốc tế FOCUS 1.x.
- **Thực thi lệnh:**
  ```bash
  python missions/m4_allocation.py
  ```
- **Nguyên lý trong code (`finops/allocation.py`):**
  - Tính tổng chi phí phát sinh theo từng tag.
  - Tính tỷ lệ dòng có đủ tag `team` và `project` hợp lệ.
  - Kiểm tra `tag_coverage >= 0.80` $\rightarrow$ Mở cổng `chargeback_ready = True`.
  - Xuất 50 dòng mẫu ra `outputs/focus_export.csv`.
- **Kết quả thu được:**
  - Tag Coverage đạt **92%** ($> 80\%$) $\rightarrow$ Đủ điều kiện triển khai Chargeback.
  - File `outputs/focus_export.csv` được tạo thành công.

---

### Bước 5: Mission 5 — Báo cáo tổng hợp & Biểu đồ Waterfall (Optimization Report)

- **Mục tiêu:** Tổng hợp kết quả từ M1 đến M4, tạo báo cáo tổng thể `outputs/report.md`, vẽ biểu đồ phân rã tiết kiệm `outputs/savings.png`, và đưa ra khuyến nghị Green AI.
- **Thực thi lệnh:**
  ```bash
  python missions/m5_report.py
  ```
- **Bốn đòn bẩy tiết kiệm được tổng hợp:**
  1. **Inference (cascade/cache/batch):** ~$1,121/tháng
  2. **Purchasing (spot/reserved):** ~$17,578/tháng
  3. **Right-size util-lies:** ~$1,519/tháng (Hạ cấp `gpu-h100-4` từ H100 về A100 và `gpu-a10g-1` từ A10G về L4)
  4. **Kill idle GPUs:** ~$600/tháng (Tắt GPU nhàn rỗi)
- **Kết quả thu được:**
  - Tổng chi phí Baseline: **$46,357/tháng**.
  - Chi phí sau tối ưu hóa: **$25,539/tháng**.
  - Tổng mức tiết kiệm: **$20,818/tháng (46.1% ≈ 46%)**.
  - File tạo ra: `outputs/report.md` và `outputs/savings.png`.

---

## 5. HƯỚNG DẪN THỰC HIỆN PHẦN MỞ RỘNG "YOUR TURN" (20 ĐIỂM)

Theo Rubric, bạn cần chọn và thực hiện **ít nhất 2 trong 5** phần mở rộng để đạt trọn vẹn 20 điểm phần D. Dưới đây là hướng dẫn chi tiết từng bài toán:

---

### Extension 1: Tinh chỉnh chính sách `recommend_tier()`
- **Vấn đề:** Hàm `recommend_tier()` mặc định chỉ xét duty cycle đơn giản và gán cứng chiết khấu 45%.
- **Cách cải tiến:**
  1. Bổ sung tham số `gpu_type` và `job_days`.
  2. Phân biệt tỷ lệ thu hồi (interruption rate): GPU dòng cũ (như A10G) có tỷ lệ thu hồi cao hơn GPU H100.
  3. So sánh hiệu quả tài chính giữa Reserved 1 năm (chiết khấu ~20-30%) vs 3 năm (chiết khấu 40-45%) dựa trên tổng số ngày dự án (`job_days`).
- **File cần chỉnh:** `finops/pricing.py` và chạy lại `missions/m3_purchasing.py` để đo lường % savings thay đổi.

---

### Extension 2: Right-sizing theo MBU cho Memory-bound Workloads
- **Vấn đề:** Một số GPU phục vụ suy luận có MFU rất thấp vì chúng bị nghẽn băng thông bộ nhớ (Decode phase).
- **Cách cải tiến:**
  1. Trong `missions/m1_efficiency_audit.py`, tính chỉ số đơn giá theo dung lượng bộ nhớ `$/GB-VRAM` và `$/TBps-Bandwidth` từ `price_catalog.csv`.
  2. Với các workload suy luận có `mbu` thấp hoặc chạy trên GPU quá đắt (như H100 cho tác vụ nhẹ), đề xuất chuyển sang L4 hoặc A10G.
  3. Báo cáo số tiền tiết kiệm hàng tháng từ việc right-sizing toàn diện.

---

### Extension 3: Điểm hòa vốn của Prompt Caching (`cache_is_worth_it`)
- **Vấn đề:** Việc ghi prompt cache vào bộ nhớ GPU/Cloud có thể phát sinh chi phí lưu trữ hoặc chi phí ghi lần đầu (Cache Write). Nếu một prompt chỉ được gọi 1 lần thì việc bật cache sẽ làm tốn thêm tiền.
- **Cách thực hiện:**
  1. Viết hàm `cache_is_worth_it(avg_cache_reads, write_cost_per_m, read_discount=0.10)` trong `finops/pricing.py`.
  2. Xác định điểm hòa vốn: Số lần đọc lại tối thiểu $N$ để tổng tiền tiết kiệm $> \text{Chi phí ghi cache}$.
  3. Tích hợp hàm này vào `missions/m2_inference_levers.py`: Chỉ áp dụng chiết khấu cache khi `cache_is_worth_it == True`.

---

### Extension 4: Quản trị & Phân bổ ngân sách Reasoning Traffic
- **Vấn đề:** Các truy vấn Reasoning (`is_reasoning=1`) sinh lượng output token rất lớn và tiêu thụ năng lượng gấp ~80 lần query chuẩn.
- **Cách thực hiện:**
  1. Trong `missions/m2_inference_levers.py`, tách riêng thống kê chi phí $ và năng lượng `Wh` cho 2 nhóm: `is_reasoning=1` vs `is_reasoning=0`.
  2. Đưa ra phân tích: Reasoning chiếm bao nhiêu % lượng truy vấn nhưng đóng góp bao nhiêu % tổng chi phí và năng lượng.
  3. Đề xuất quy tắc định tuyến: Chỉ kích hoạt Reasoning cho các bài toán phức tạp (như code/toán) hoặc khi mức độ tự tin của model nhỏ < 0.7.

---

### Extension 5: Lập lịch huấn luyện nhận thức Carbon (Carbon-aware Scheduling)
- **Vấn đề:** Các job huấn luyện (Training batch) không cần chạy gấp theo thời gian thực có thể được điều phối chạy ở các datacenter xanh.
- **Cách thực hiện:**
  1. Dùng bảng `REGION_CARBON` và `REGION_PRICE_KWH` trong `finops/sustainability.py`.
  2. Tính toán lượng carbon phát thải của các job `interruptible=1` nếu chạy ở vùng mặc định (`us-east-1`: 380 gCO2/kWh) so với vùng sạch nhất (`europe-north1`: 30 gCO2/kWh).
  3. In ra số kg CO2e giảm được và phân tích trade-off về độ trễ mạng (Network Latency).

---

## 6. HƯỚNG DẪN KIỂM THỬ, XÁC MINH & ĐÁNH GIÁ (RUBRIC)

### 6.1. Xác minh tự động qua `verify.py` (30 điểm)

Chạy lệnh kiểm tra tổng thể:
```bash
python verify.py
```

**Bảng 11 tiêu chí tự động cần đạt (11/11 PASS):**
1. `M1 flags the GPU-Util lie (gpu-h100-4)`: Nhận diện chính xác GPU H100 chạy giả hiệu quả.
2. `M1 detects idle waste`: Tính toán được lãng phí chạy không tải (> $0/ngày).
3. `M2 $/1M-token drops after optimization`: Đơn giá $/1M-token sau tối ưu phải thấp hơn trước.
4. `M2 inference savings in 60-95% band`: Tỷ lệ tiết kiệm suy luận nằm trong khoảng 60% – 95%.
5. `M3 recommends a spot tier`: Có đề xuất ít nhất một tác vụ dùng Spot.
6. `M3 recommends a reserved tier`: Có đề xuất ít nhất một tác vụ dùng Reserved.
7. `M3 purchasing saves money`: Mua sắm có chiến lược tiết kiệm được tiền (> 0%).
8. `M4 tag coverage 85-100%`: Tỷ lệ gắn tag đạt chuẩn (85% – 100%).
9. `M4 chargeback gate is open`: Cổng Chargeback được mở (`chargeback_ready == True`).
10. `M5 total savings in 40-95% band`: Tổng mức cắt giảm chi phí toàn diện đạt 40% – 95%.
11. `M5 report.md written`: File báo cáo Markdown được ghi thành công vào thư mục `outputs/`.

---

### 6.2. Chạy bộ Unit Tests với `pytest` (20 điểm)

Chạy 15 bài kiểm thử đơn vị và kiểm thử tích hợp:
```bash
pytest -q
```
**Kết quả mong đợi:** `15 passed in 0.xx s`.

> **LƯU Ý QUAN TRỌNG:** Tuyệt đối không chỉnh sửa các file trong thư mục `tests/` để hardcode kết quả. Mọi logic phải được giải quyết chuẩn xác trong thư mục `finops/` và `missions/`.

---

### 6.3. Tiêu chí chấm Báo cáo kỹ thuật (30 điểm) & Phần mở rộng (20 điểm)

- **Báo cáo kỹ thuật (30 điểm):**
  - Đầy đủ số liệu so sánh Baseline vs Optimized theo $/tháng và $/1M-token.
  - Bảng phân rã chi tiết 4 đòn bẩy tiết kiệm.
  - Mục Sustainability phân tích rõ năng lượng, carbon và vị trí triển khai.
  - Giải thích sâu sắc nguyên nhân kỹ thuật của "GPU-Util Lie".
  - Có file biểu đồ `outputs/savings.png` hiển thị trực quan.
- **Phần mở rộng Your Turn (20 điểm):**
  - Thực hiện thành công $\ge 2$ extensions với mã nguồn hoàn chỉnh, có in kết quả định lượng so sánh trước/sau và rút ra bài học kinh nghiệm.

---

## 7. HƯỚNG DẪN VIẾT BÁO CÁO WRITE-UP & CHUẨN BỊ NỘP BÀI

Để đạt điểm tối đa (100/100), bạn nên chuẩn bị một bài viết giải trình (Write-up) ngắn gọn (1–2 trang Markdown hoặc PDF) trả lời 5 câu hỏi cốt lõi sau:

### Cấu trúc bài Write-up đề xuất:

1. **Tổng kết Baseline vs. Optimized:**
   - Chi phí trước tối ưu: `$46,357/tháng` ($\$6.49/1\text{M-token}$).
   - Chi phí sau tối ưu: `$25,539/tháng` ($\$1.13/1\text{M-token}$).
   - Tổng mức tiết kiệm: `$20,818/tháng` (**46.1%**).
2. **Phân tích Hiệu quả từng Đòn bẩy:**
   - Đòn bẩy nào đóng góp số tiền tuyệt đối lớn nhất? (Purchasing Strategy đóng góp ~$17,578/tháng).
   - Đòn bẩy nào có tỷ lệ cắt giảm % cao nhất? (Inference Levers cắt giảm ~82.6% chi phí gọi LLM nhờ Discount Stacking).
3. **Bài học từ hiện tượng "GPU-Util Lie":**
   - Phân tích trường hợp `gpu-h100-4` (Util 98% nhưng MFU 20.2%).
   - Giải thích cơ chế vì sao `nvidia-smi` đánh lừa kỹ sư và tại sao MFU mới là thước đo phản ánh đúng chi phí.
4. **Trình bày các phần mở rộng đã thực hiện (Your Turn Extensions):**
   - Trình bày chi tiết 2 extension bạn đã chọn làm.
   - Nêu rõ các con số đo lường trước và sau khi cải tiến.
5. **Đề xuất Hành động cho Ban Giám đốc NimbusAI (Action Plan):**
   - *Hành động 1 (Ngay lập tức):* Bật Prompt Caching và định tuyến Model Cascading cho toàn bộ API Gateway.
   - *Hành động 2 (Trong tuần 1):* Tắt các GPU nhàn rỗi qua đêm và chuyển các job huấn luyện ngắt quãng sang Spot Instances có Checkpoint.
   - *Hành động 3 (Trong tháng 1):* Ban hành chính sách Tagging bắt buộc để duy trì Tag Coverage > 80% và kích hoạt hệ thống Chargeback nội bộ.

### Các file cần nộp lên hệ thống:
```
outputs/report.md
outputs/savings.png
outputs/focus_export.csv
[Bài viết giải trình Write-up .md hoặc .pdf]
```

---

## 8. CÁC LỖI THƯỜNG GẶP & CÁCH XỬ LÝ (TROUBLESHOOTING)

1. **Lỗi `ModuleNotFoundError: No module named 'pandas'` hoặc `'matplotlib'`:**
   - *Nguyên nhân:* Chưa kích hoạt môi trường ảo `.venv` hoặc chưa cài đặt `requirements.txt`.
   - *Khắc phục:* Kích hoạt lại `.venv` (`.venv\Scripts\activate` trên Windows) và chạy `pip install -r requirements.txt`.
2. **Lỗi `FileNotFoundError: data/gpu_telemetry.csv`:**
   - *Nguyên nhân:* Chưa chạy script sinh dữ liệu ban đầu.
   - *Khắc phục:* Chạy `python data/generate.py`.
3. **Lỗi `verify.py` báo `M2 inference savings out of band`:**
   - *Nguyên nhân:* Sai tỷ lệ chiết khấu trong `finops/pricing.py`.
   - *Khắc phục:* Đảm bảo `cache_discount = 0.10` (giảm 90%) và `batch_discount = 0.50` (giảm 50%).
4. **Lỗi `pytest` fail `test_flag_util_lies`:**
   - *Nguyên nhân:* Hàm `flag_util_lies()` nhận `gpu_util_pct` thang 0–100 nhưng so sánh trực tiếp không chia 100.
   - *Khắc phục:* Kiểm tra `util = float(r.get("gpu_util_pct", 0)) / 100.0` với điều kiện `util >= 0.90` và `mfu < 0.30`.
5. **Cảnh báo liên quan đến `is_batch` (Boolean conversion):**
   - *Khắc phục:* Ép kiểu an toàn bằng `is_batch = bool(int(num(r["is_batch"])))`.

---
*Tài liệu được biên soạn nhằm hỗ trợ sinh viên hoàn thành xuất sắc Lab 25 - GPU FinOps Optimization.*
