# Telegram Compact RandomForest Demo - Clean Version

Bản này đã bỏ toàn bộ phần inference thử và các hướng dẫn dài trong giao diện Streamlit.

Dashboard chỉ giữ các phần cần demo kết quả:

1. Overview
2. Channel Cluster Ranking
3. RandomForest Model Explanation
4. SVD & Network Evidence
5. Network Cell Logs
6. Raw Results

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy

```bash
streamlit run app.py
```

## Nội dung chính

- Ranking channel clusters theo amplification score.
- So sánh observed amplification với RandomForest score.
- Feature importance của RandomForest.
- Biểu đồ SVD và Network từ output notebook.
- Log cell từ notebook network.


## Bổ sung v6

- Đã bỏ phần `Xem full/truncated log`.
- Tab `Network Cell Logs` chỉ còn bảng preview log.
- Các thống kê từ `build-network` đã được phóng to thành card lớn:
  - Network edges
  - Channel profiles
  - Communities
  - Parquet errors
- Overview cũng hiển thị thống kê build-network bằng card lớn.


## Bổ sung v7

- Các ảnh trong phần `Network evidence` đã được phóng to.
- Riêng biểu đồ `Feature distributions by channel archetype` hiển thị full-width để dễ đọc hơn.
- SVD vẫn giữ dạng 2 cột gọn.


## Bổ sung v8

- Sửa lỗi `NameError: big_build_metric is not defined`.
- Hàm `big_build_metric()` đã được định nghĩa trước khi gọi trong Overview và Network Cell Logs.
