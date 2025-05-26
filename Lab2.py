# =============================================================================
# STREAMLIT IMAGE PROCESSING APP
# =============================================================================
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# =============================================================================
# CẤU HÌNH TRANG
# =============================================================================
st.set_page_config(
    page_title="Xử lý ảnh với Streamlit",
    page_icon="🖼️",
    layout="wide"
)

# =============================================================================
# KHỞI TẠO SESSION STATE
# =============================================================================
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

# =============================================================================
# TIÊU ĐỀ VÀ SIDEBAR
# =============================================================================
st.title("🖼️ Ứng dụng Xử lý Ảnh")
st.markdown("---")

# Sidebar cho các điều khiển
st.sidebar.title("Điều khiển")

# =============================================================================
# PHẦN 1: TẢI ẢNH LÊN
# =============================================================================
st.sidebar.subheader("📁 Tải ảnh")
uploaded_file = st.sidebar.file_uploader(
    "Chọn ảnh của bạn", 
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    # Đọc và chuyển đổi ảnh
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    
    # Nếu ảnh có 4 kênh (RGBA), chuyển về 3 kênh (RGB)
    if len(image_array.shape) == 3 and image_array.shape[2] == 4:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
        pass  # Đã là RGB
    else:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    
    # Lưu vào session state
    st.session_state.original_image = image_array
    st.session_state.current_image = image_array.copy()

# Nút reset
if st.sidebar.button("🔄 Reset ảnh"):
    if st.session_state.original_image is not None:
        st.session_state.current_image = st.session_state.original_image.copy()

# =============================================================================
# PHẦN 2: CÂN BẰNG MÀU
# =============================================================================
st.sidebar.subheader("🎨 1. Cân bằng màu")
st.sidebar.markdown("Điều chỉnh độ sáng từng kênh màu (0.0-2.0)")

red_factor = st.sidebar.slider("Red", 0.0, 2.0, 1.0, 0.1)
green_factor = st.sidebar.slider("Green", 0.0, 2.0, 1.0, 0.1)
blue_factor = st.sidebar.slider("Blue", 0.0, 2.0, 1.0, 0.1)

if st.sidebar.button("Áp dụng cân bằng màu"):
    if st.session_state.current_image is not None:
        img = st.session_state.current_image.astype(np.float32)
        img[:,:,0] *= red_factor   # Red
        img[:,:,1] *= green_factor # Green
        img[:,:,2] *= blue_factor  # Blue
        st.session_state.current_image = np.clip(img, 0, 255).astype(np.uint8)

# =============================================================================
# PHẦN 3: HISTOGRAM
# =============================================================================
st.sidebar.subheader("📊 2. Histogram")

if st.sidebar.button("Hiển thị histogram"):
    if st.session_state.current_image is not None:
        st.session_state.show_histogram = True

if st.sidebar.button("Cân bằng histogram"):
    if st.session_state.current_image is not None:
        yuv = cv2.cvtColor(st.session_state.current_image, cv2.COLOR_RGB2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        st.session_state.current_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

# =============================================================================
# PHẦN 4: BỘ LỌC
# =============================================================================
st.sidebar.subheader("🔧 3-4. Bộ lọc")

kernel_size = st.sidebar.selectbox(
    "Chọn kernel size", 
    [3, 5, 7, 9, 11], 
    index=1
)

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Lọc trung vị"):
        if st.session_state.current_image is not None:
            for i in range(3):
                st.session_state.current_image[:,:,i] = cv2.medianBlur(
                    st.session_state.current_image[:,:,i], kernel_size
                )

with col2:
    if st.button("Lọc trung bình"):
        if st.session_state.current_image is not None:
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            st.session_state.current_image = cv2.filter2D(
                st.session_state.current_image, -1, kernel
            )

# =============================================================================
# PHẦN 5: CHỨC NĂNG PHỤ
# =============================================================================
st.sidebar.subheader("🎲 Test")

if st.sidebar.button("Thêm nhiễu"):
    if st.session_state.current_image is not None:
        noise = np.random.random(st.session_state.current_image.shape[:2])
        st.session_state.current_image[noise < 0.05] = 0
        st.session_state.current_image[noise > 0.95] = 255

# =============================================================================
# HIỂN THỊ ẢNH CHÍNH
# =============================================================================
if st.session_state.current_image is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh gốc")
        st.image(st.session_state.original_image, use_column_width=True)
    
    with col2:
        st.subheader("Ảnh đã xử lý")
        st.image(st.session_state.current_image, use_column_width=True)
    
    # Thông tin ảnh
    h, w, c = st.session_state.current_image.shape
    st.info(f"📐 Kích thước: {w} x {h} pixels | 🎨 Kênh màu: {c}")
    
    # =============================================================================
    # HIỂN THỊ HISTOGRAM
    # =============================================================================
    if hasattr(st.session_state, 'show_histogram') and st.session_state.show_histogram:
        st.subheader("📊 Histogram RGB")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['red', 'green', 'blue']
        
        for i, color in enumerate(colors):
            hist = cv2.calcHist([st.session_state.current_image], [i], None, [256], [0, 256])
            ax.plot(hist, color=color, alpha=0.7, label=f'{color.capitalize()} channel')
        
        ax.set_title('Histogram RGB')
        ax.set_xlabel('Giá trị pixel')
        ax.set_ylabel('Tần suất')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close()
        
        # Reset flag
        st.session_state.show_histogram = False
    
    # =============================================================================
    # TẢI XUỐNG ẢNH
    # =============================================================================

else:
    # Hiển thị hướng dẫn khi chưa có ảnh
    st.info("👆 Hãy tải ảnh lên từ sidebar để bắt đầu xử lý!")
    
    st.markdown("""
    ### 🚀 Hướng dẫn sử dụng:
    
    1. **📁 Tải ảnh**: Sử dụng file uploader ở sidebar
    2. **🎨 Cân bằng màu**: Điều chỉnh độ sáng từng kênh màu RGB
    3. **📊 Histogram**: Xem phân bố màu và cân bằng histogram
    4. **🔧 Bộ lọc**: Áp dụng lọc trung vị hoặc trung bình để giảm nhiễu
    5. **🎲 Test**: Thêm nhiễu để test các bộ lọc
    """)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🖼️ <strong>Image Processing App</strong> | Phát triển bởi Đỗ Quốc Dũng</p>
    </div>
    """, 
    unsafe_allow_html=True
)