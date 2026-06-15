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


