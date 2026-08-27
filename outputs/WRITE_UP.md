# BÁO CÁO GIẢI TRÌNH KỸ THUẬT: GPU FINOPS OPTIMIZATION (NIMBUSAI)

> **Học viên:** Nguyễn Hoàng Việt
> **Mã học viên:** 2A202601940

---

## 1. TỔNG QUAN HIỆU QUẢ TỐI ƯU: BASELINE VS. OPTIMIZED

Sau khi phân tích toàn diện 11 GPU, 8 workloads huấn luyện/suy luận và 2,400 lượt gọi API LLM thực tế tại NimbusAI, hệ thống FinOps đã mang lại kết quả cắt giảm chi phí ấn tượng:

| Chỉ số | Baseline (Trước tối ưu) | Optimized (Sau tối ưu) | Mức cắt giảm |
|---|---|---|---|
| **Tổng chi phí vận hành** | **$27,133 / tháng** | **$14,626 / tháng** | **-$12,507 / tháng (46.1%)** |
| **Đơn giá suy luận (`$/1M-token`)** | **$6.488 / 1M-token** | **$1.126 / 1M-token** | **-82.6%** |
| **Chi phí mua sắm hạ tầng (`$/tháng`)**| **$25,667 / tháng** | **$15,627 / tháng** | **-$10,040 / tháng (39.1%)** |
| **Lãng phí chạy không tải (Idle)** | **$600 / tháng** | **$0 / tháng** | **-100.0% ($20/ngày)** |

---

## 2. PHÂN TÍCH ĐÓNG GÓP CỦA 4 ĐÒN BẨY FINOPS (SAVINGS WATERFALL)

Dựa trên biểu đồ `outputs/savings.png` và báo cáo `outputs/report.md`, tổng mức tiết kiệm **$12,507 / tháng** được phân rã cụ thể:

```
[Baseline: $27,133]
  ├── - $10,040 (80.3%) : Purchasing Strategy (Spot + Checkpoint & Reserved 3yr)
  ├── - $ 1,212 ( 9.7%) : Inference Levers (Cascade + 90% Prompt Cache + 50% Batch API)
  ├── - $   655 ( 5.2%) : Right-sizing "GPU-Util Lies" (Hạ cấp H100->A100, A10G->L4)
  └── - $   600 ( 4.8%) : Loại bỏ GPU Idle qua đêm (Kill idle H100-5)
[Optimized: $14,626]
```

### Đánh giá trọng số từng đòn bẩy:
1. **Purchasing Strategy (Đóng góp lớn nhất về số tiền tuyệt đối - $10,040/tháng):** Chuyển 5 job training/eval gián đoạn sang Spot Instance và cam kết Reserved 3 năm cho 3 job suy luận liên tục 24/7 (Duty cycle = 100% > 55% Break-even).
2. **Inference Levers (Đóng góp tỷ lệ % cắt giảm cao nhất - 82.6% trên tầng suy luận):** Định tuyến 80% truy vấn đơn giản sang model nhỏ ($0.20/$0.40), kết hợp Prompt Caching (giảm 90% input) và Batch API (giảm 50%). Khi kết hợp cả 2, discount stack đạt `0.050` (tiết kiệm tới 95% đơn giá gốc).

---

## 3. BÓC TRẦN HIỆN TƯỢNG "GPU-UTIL LIE"

Trong quá trình kiểm toán M1, hệ thống đã phát hiện 2 GPU có hiện tượng "nói dối" nghiêm trọng:
* **`gpu-h100-4` (H100):** `GPU-Util = 98.2%` nhưng **MFU chỉ đạt 0.194 (19.4%)** và MBU đạt 0.207.
* **`gpu-a10g-1` (A10G):** `GPU-Util = 96.9%` nhưng **MFU chỉ đạt 0.268 (26.8%)**.

### Bản chất kỹ thuật:
`nvidia-smi` chỉ đo tỷ lệ thời gian clock của GPU có xung nhịp hoạt động. Nó **không đo lượng phép tính ma trận FLOPs thực tế**. 
Khi mã nguồn bị nghẽn nạp dữ liệu từ RAM/HBM (Memory Stall), kernel launch overhead quá lớn, hoặc batch size quá nhỏ (Batch=1), nhân CUDA/Tensor Core phải chờ dữ liệu nhưng clock vẫn báo bận. Doanh nghiệp phải trả $2.50/giờ cho H100 nhưng chỉ nhận được hiệu năng của một GPU $0.80/giờ.

**Giải pháp:** Hạ cấp `gpu-h100-4` về A100 và `gpu-a10g-1` về L4, tiết kiệm ngay **$655/tháng**.

---

## 4. KẾT QUẢ ĐO LƯỜNG CÁC PHẦN MỞ RỘNG (YOUR TURN EXTENSIONS)

### Extension 3: Kinh tế học Prompt Caching (`cache_is_worth_it`)
* **Dữ liệu thực tế:** Trong 2,400 request, 100% request có phần prompt dùng chung; tổng input tokens là 5,343,437 token, trong đó có **1,703,990 cached tokens (31.9%)**.
* **Điểm hòa vốn:** Với giá ghi cache \$3.75/1M token và chiết khấu đọc 90% (\$0.30/1M token), điểm hòa vốn là:
  $$\text{Break-even} = \frac{\$3.75}{\$3.00 \times (1 - 0.10)} = 1.39 \text{ lượt đọc}$$
* **Kết luận:** Với tần suất đọc lại trung bình $\ge 2$ lần/prefix, Prompt Caching mang lại lợi nhuận ròng dương (`cache_is_worth_it == True`).

### Extension 4: Phân tích & Quản trị Ngân sách Reasoning Traffic
* **Dữ liệu đo đạc:**
  * Truy vấn chuẩn: 2,199 request (91.6% volume) tiêu tốn \$7.09/ngày (83.5% chi phí) và 1,887.6 Wh (6.0% điện năng).
  * Truy vấn Reasoning (`is_reasoning=1`): Chỉ 201 request (**8.4% volume**) nhưng tiêu tốn \$1.40/ngày (16.5% chi phí) và **29,787.7 Wh (94.0% điện năng)**.
* **Hệ số năng lượng:** Mỗi query Reasoning tiêu thụ trung bình **148.20 Wh/query**, gấp **~172.6 lần** so với query chuẩn (0.86 Wh/query).
* **Đề xuất chính sách:** Thiết lập bộ lọc Gateway chỉ kích hoạt Reasoning cho bài toán chấm code/toán học hoặc khi Model Confidence Score < 0.70.

### Extension 5: Lập lịch huấn luyện nhận thức Carbon (Carbon-aware Scheduling)
* **Dữ liệu đo đạc:** 5 job training gián đoạn tiêu thụ **4,227.0 kWh điện / tháng**.
* **So sánh vùng:**
  * `us-east-1` (Vùng mặc định): Phát thải 1,606.3 kg CO2e/tháng, tiền điện \$507.24.
  * `europe-north1` (Na Uy - Thủy điện): Phát thải **126.8 kg CO2e/tháng**, tiền điện \$380.43.
* **Kết quả:** Chuyển dịch job training sang Na Uy giúp **giảm 1,479.5 kg CO2e/tháng (giảm 92.1% phát thải)** và tiết kiệm thêm \$126.81 tiền điện mỗi tháng.

---

## 5. KHUYẾN NGHỊ HÀNH ĐỘNG CHO BAN GIÁM ĐỐC NIMBUSAI

1. **Tuần 1 (Quick Wins - Không tốn chi phí triển khai):**
   * Kích hoạt Prompt Caching và định tuyến Model Cascading trên API Gateway.
   * Cấu hình Auto-shutdown tắt GPU `gpu-h100-5` khi idle quá 30 phút.
2. **Tháng 1 (Tối ưu hóa Hạ tầng Mua sắm):**
   * Chuyển 5 workloads huấn luyện sang Spot Instance kết hợp lưu Checkpoint mỗi 30 phút.
   * Ký hợp đồng Reserved 3 năm cho 3 cụm GPU phục vụ Chat/RAG/Search cố định.
3. **Quý 1 (Quản trị & Bền vững):**
   * Triển khai xuất hóa đơn FOCUS 1.x định kỳ và áp dụng Chargeback nội bộ (Tag coverage hiện tại đạt 92% > 80%).
   * Thiết lập cơ chế Carbon-aware Dispatcher tự động đẩy batch job huấn luyện sang vùng `europe-north1`.

---
*Báo cáo được trích xuất tự động từ hệ thống đo lường FinOps chuẩn xác của Lab 25.*
