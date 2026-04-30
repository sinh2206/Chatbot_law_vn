from datasets import load_dataset
import pandas as pd

# 1. Danh sách 17 số hiệu văn bản bạn cần lấy
danh_sach_so_hieu = [
    "59/2020/QH14", "168/2025/NĐ-CP", "60/2014/QH13", "123/2015/NĐ-CP",
    "87/2020/NĐ-CP", "04/2020/TT-BTP", "26/2023/QH15", "59/2022/NĐ-CP",
    "31/2024/QH15", "43/2014/NĐ-CP", "148/2020/NĐ-CP", "38/2019/QH14",
    "126/2020/NĐ-CP", "91/2022/NĐ-CP", "80/2021/TT-BTC", "40/2025/TT-BTC",
    "86/2024/TT-BTC"
]

# ==========================================
# BƯỚC 1: TẢI METADATA VÀ TÌM ID
# ==========================================
print("1. Đang tải tập METADATA (nhẹ)...")
ds_metadata = load_dataset("th1nhng0/vietnamese-legal-documents", "metadata", split="data")

def loc_metadata(dong):
    so_hieu = str(dong.get('document_number', '')).strip()
    return so_hieu in danh_sach_so_hieu

print("   Đang tìm ID dựa trên số hiệu...")
metadata_da_loc = ds_metadata.filter(loc_metadata)

# Rút trích danh sách ID ra thành một list (mảng)
danh_sach_id = metadata_da_loc['id']
print(f"-> Tuyệt vời! Đã tìm thấy {len(danh_sach_id)} ID tương ứng: {danh_sach_id}")

# ==========================================
# BƯỚC 2: TẢI CONTENT VÀ LỌC NGƯỢC THEO ID
# ==========================================
print("\n2. Đang tải tập CONTENT (nặng)...")
ds_content = load_dataset("th1nhng0/vietnamese-legal-documents", "content", split="data")

# Tạo một tập hợp (set) từ danh sách ID để code tìm kiếm nhanh hơn gấp nhiều lần
tap_hop_id = set(danh_sach_id)

def loc_content_theo_id(dong):
    # Chỉ giữ lại những dòng có id nằm trong tap_hop_id
    return dong.get('id') in tap_hop_id

print("   Đang trích xuất nội dung dựa trên ID...")
content_da_loc = ds_content.filter(loc_content_theo_id)

print(f"-> Hoàn tất! Đã lấy thành công nội dung của {len(content_da_loc)} văn bản.")

# ==========================================
# BƯỚC 3: XUẤT DỮ LIỆU RA FILE
# ==========================================
# Chuyển đổi sang Pandas DataFrame
df = content_da_loc.to_pandas()

# Lưu ra file CSV (Hỗ trợ tiếng Việt UTF-8)
ten_file = "17_van_ban_phap_luat_chon_loc.csv"
df.to_csv(ten_file, index=False, encoding="utf-8-sig")
print(f"\n🎉 Đã xuất toàn bộ nội dung ra file: {ten_file}")