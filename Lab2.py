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
if 'show_histogram' not in st.session_state:
    st.session_state.show_histogram = False

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
        st.success("Đã áp dụng cân bằng màu!")

# =============================================================================
# PHẦN 3: HISTOGRAM
# =============================================================================
st.sidebar.subheader("📊 2. Histogram")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.sidebar.button("Hiển thị histogram"):
        if st.session_state.current_image is not None:
            st.session_state.show_histogram = True

with col2:
    if st.sidebar.button("Cân bằng histogram"):
        if st.session_state.current_image is not None:
            # Chuyển sang không gian màu YUV
            yuv = cv2.cvtColor(st.session_state.current_image, cv2.COLOR_RGB2YUV)
            # Cân bằng histogram cho kênh Y (luminance)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            # Chuyển lại RGB
            st.session_state.current_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
            st.success("Đã cân bằng histogram!")

# =============================================================================
# PHẦN 4: BỘ LỌC
# =============================================================================
st.sidebar.subheader("🔧 3-4. Bộ lọc")

kernel_size = st.sidebar.selectbox(
    "Chọn kernel size", 
    [3, 5, 7, 9, 11], 
    index=1
)

# Tham số cho Gaussian blur
sigma_x = st.sidebar.slider("Sigma X (Gaussian)", 0.1, 5.0, 1.0, 0.1)
sigma_y = st.sidebar.slider("Sigma Y (Gaussian)", 0.1, 5.0, 1.0, 0.1)

# Bố trí các nút lọc
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Lọc trung vị"):
        if st.session_state.current_image is not None:
            # Áp dụng median filter cho từng kênh màu
            filtered_img = st.session_state.current_image.copy()
            for i in range(3):
                filtered_img[:,:,i] = cv2.medianBlur(
                    st.session_state.current_image[:,:,i], kernel_size
                )
            st.session_state.current_image = filtered_img
            st.success("Đã áp dụng lọc trung vị!")

with col2:
    if st.button("Lọc trung bình"):
        if st.session_state.current_image is not None:
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            st.session_state.current_image = cv2.filter2D(
                st.session_state.current_image, -1, kernel
            )
            st.success("Đã áp dụng lọc trung bình!")

# Thêm nút Gaussian blur
col3, col4 = st.sidebar.columns(2)
with col3:
    if st.button("Lọc Gaussian"):
        if st.session_state.current_image is not None:
            # Đảm bảo kernel size là số lẻ
            k_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            st.session_state.current_image = cv2.GaussianBlur(
                st.session_state.current_image, 
                (k_size, k_size), 
                sigmaX=sigma_x, 
                sigmaY=sigma_y
            )
            st.success("Đã áp dụng lọc Gaussian!")

with col4:
    if st.button("Khử Gaussian"):
        if st.session_state.current_image is not None:
            # Tạo bản sao làm mờ
            k_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            blurred = cv2.GaussianBlur(
                st.session_state.original_image, 
                (k_size, k_size), 
                sigmaX=sigma_x, 
                sigmaY=sigma_y
            )
            
            # Unsharp masking: original + alpha * (original - blurred)
            alpha = 1.5  # Hệ số tăng cường độ sắc nét
            unsharp_mask = st.session_state.original_image.astype(np.float32) + \
                          alpha * (st.session_state.original_image.astype(np.float32) - blurred.astype(np.float32))
            
            st.session_state.current_image = np.clip(unsharp_mask, 0, 255).astype(np.uint8)
            st.success("Đã áp dụng khử Gaussian (Unsharp Masking)!")

# =============================================================================
# PHẦN 5: CHỨC NĂNG PHỤ
# =============================================================================
st.sidebar.subheader("🎲 Test")

# Thêm tham số cho nhiễu
noise_intensity = st.sidebar.slider("Cường độ nhiễu", 0.01, 0.2, 0.05, 0.01)

if st.sidebar.button("Thêm nhiễu salt & pepper"):
    if st.session_state.current_image is not None:
        noise = np.random.random(st.session_state.current_image.shape[:2])
        noisy_img = st.session_state.current_image.copy()
        noisy_img[noise < noise_intensity/2] = 0      # Salt (đen)
        noisy_img[noise > 1 - noise_intensity/2] = 255  # Pepper (trắng)
        st.session_state.current_image = noisy_img
        st.success("Đã thêm nhiễu salt & pepper!")

if st.sidebar.button("Thêm nhiễu Gaussian"):
    if st.session_state.current_image is not None:
        # Thêm nhiễu Gaussian
        noise = np.random.normal(0, 25, st.session_state.current_image.shape).astype(np.float32)
        noisy_img = st.session_state.current_image.astype(np.float32) + noise
        st.session_state.current_image = np.clip(noisy_img, 0, 255).astype(np.uint8)
        st.success("Đã thêm nhiễu Gaussian!")

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
    if st.session_state.show_histogram:
        st.subheader("📊 Histogram RGB")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        colors = ['red', 'green', 'blue']
        
        # Histogram ảnh gốc
        for i, color in enumerate(colors):
            hist = cv2.calcHist([st.session_state.original_image], [i], None, [256], [0, 256])
            ax1.plot(hist, color=color, alpha=0.7, label=f'{color.capitalize()} channel')
        
        ax1.set_title('Histogram - Ảnh gốc')
        ax1.set_xlabel('Giá trị pixel')
        ax1.set_ylabel('Tần suất')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Histogram ảnh đã xử lý
        for i, color in enumerate(colors):
            hist = cv2.calcHist([st.session_state.current_image], [i], None, [256], [0, 256])
            ax2.plot(hist, color=color, alpha=0.7, label=f'{color.capitalize()} channel')
        
        ax2.set_title('Histogram - Ảnh đã xử lý')
        ax2.set_xlabel('Giá trị pixel')
        ax2.set_ylabel('Tần suất')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Nút ẩn histogram
        if st.button("Ẩn histogram"):
            st.session_state.show_histogram = False
    
    # =============================================================================
    # TẢI XUỐNG ẢNH
    # =============================================================================
    st.subheader("💾 Tải xuống")
    
    # Tạo buffer cho ảnh
    img_buffer = io.BytesIO()
    pil_img = Image.fromarray(st.session_state.current_image)
    pil_img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    st.download_button(
        label="📥 Tải xuống ảnh đã xử lý",
        data=img_buffer,
        file_name="processed_image.png",
        mime="image/png"
    )

else:
    # Hiển thị hướng dẫn khi chưa có ảnh
    st.info("👆 Hãy tải ảnh lên từ sidebar để bắt đầu xử lý!")
    
    st.markdown("""
    ### 🚀 Hướng dẫn sử dụng:
    
    1. **📁 Tải ảnh**: Sử dụng file uploader ở sidebar
    2. **🎨 Cân bằng màu**: Điều chỉnh độ sáng từng kênh màu RGB
    3. **📊 Histogram**: Xem phân bố màu và cân bằng histogram
    4. **🔧 Bộ lọc**: 
       - Lọc trung vị: Giảm nhiễu salt & pepper
       - Lọc trung bình: Làm mờ ảnh
       - Lọc Gaussian: Làm mờ mịn hơn
       - Khử Gaussian: Tăng cường độ sắc nét (Unsharp Masking)
    5. **🎲 Test**: Thêm nhiễu để test các bộ lọc
    6. **💾 Tải xuống**: Lưu ảnh đã xử lý
    
    ### 🔧 Các cải tiến mới:
    - ✅ **Khử Gaussian** với kỹ thuật Unsharp Masking
    - ✅ **Histogram so sánh** giữa ảnh gốc và ảnh xử lý
    - ✅ **Nhiễu Gaussian** bổ sung
    - ✅ **Thông báo trạng thái** khi áp dụng filter
    - ✅ **Tải xuống ảnh** đã xử lý
    """)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🖼️ <strong>Image Processing App</strong> | Phát triển bởi Đỗ Quốc Dũng</p>
        <p><em>Phiên bản cải tiến với Gaussian Blur/Unsharp và Histogram cải thiện</em></p>
    </div>
    """, 
    unsafe_allow_html=True
)
