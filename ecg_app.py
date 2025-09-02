import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import median_filter
from skimage.morphology import thin
from scipy.signal import savgol_filter
import io
import base64

# Set page config
st.set_page_config(
    page_title="ECG Digitization Web App",
    layout="wide"
)

def process_ecg_image(uploaded_image):
    """Process the uploaded ECG image and extract lead data"""
    
    # Step 1: Convert PIL image to grayscale
    image = uploaded_image.convert('L')
    image_np = np.array(image)
    
    # Step 2: Apply binarization
    _, binary_image = cv2.threshold(image_np, 50, 255, cv2.THRESH_BINARY)
    
    # Step 3: Signal enhancement
    # Median filtering to remove noise
    denoised = median_filter(binary_image, size=3)
    
    # Invert image for morphological operations
    inverted = cv2.bitwise_not(denoised)
    
    # Mild dilation to bridge gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(inverted, kernel, iterations=1)
    
    # Prepare binary mask for thinning
    binary_mask = (dilated // 255).astype(np.uint8)
    
    # Apply thinning
    thinned = thin(binary_mask).astype(np.uint8) * 255
    
    # Invert back to black-on-white
    processed_image = cv2.bitwise_not(thinned)
    
    return image_np, binary_image, processed_image

def detect_baselines(processed_image):
    """Detect baseline positions for the three ECG leads"""
    
    # Convert to binary signal mask
    signal_mask = (processed_image < 128).astype(np.uint8)
    
    # Split into 3 vertical chunks
    height, width = signal_mask.shape
    split1 = signal_mask[0:height//3, :]
    split2 = signal_mask[height//3:2*height//3, :]
    split3 = signal_mask[2*height//3:, :]
    
    segments = [split1, split2, split3]
    baselines = []
    
    # Detect row with most signal in each segment
    for i, segment in enumerate(segments):
        row_sums = np.sum(segment, axis=1)
        baseline_row = np.argmax(row_sums)
        global_row = baseline_row + i * (height // 3)
        baselines.append(global_row)
    
    return segments, baselines

def digitize_leads(segments, baselines, processed_image):
    """Digitize the ECG leads from the processed image"""
    
    height, width = processed_image.shape
    digitized_leads = []
    
    for i, segment in enumerate(segments):
        baseline_local = baselines[i] - i * (height // 3)
        x_vals = []
        y_vals = []
        
        for col in range(width):
            col_data = segment[:, col]
            black_rows = np.where(col_data > 0)[0]
            
            if len(black_rows) > 0:
                furthest_row = black_rows[np.argmax(np.abs(black_rows - baseline_local))]
                voltage = baseline_local - furthest_row
                x_vals.append(col)
                y_vals.append(voltage)
            else:
                x_vals.append(col)
                y_vals.append(np.nan)
        
        digitized_leads.append((x_vals, y_vals))
    
    return digitized_leads

def smooth_leads(digitized_leads):
    """Apply Savitzky-Golay filter to smooth the leads"""
    
    smoothed_leads = []
    
    for x_vals, y_vals in digitized_leads:
        y_array = np.array(y_vals)
        
        # Handle NaNs by interpolation
        if np.isnan(y_array).any():
            not_nan = ~np.isnan(y_array)
            if np.sum(not_nan) > 1:  # Need at least 2 points for interpolation
                y_array = np.interp(np.arange(len(y_array)), np.flatnonzero(not_nan), y_array[not_nan])
        
        # Apply Savitzky-Golay filter
        if len(y_array) > 9:
            window_length = min(9, len(y_array) if len(y_array) % 2 == 1 else len(y_array) - 1)
            smoothed_y = savgol_filter(y_array, window_length=window_length, polyorder=3)
        else:
            smoothed_y = y_array  # Skip filtering for very short signals
        
        smoothed_leads.append((x_vals, smoothed_y.tolist()))
    
    return smoothed_leads

def create_plots(smoothed_leads):
    """Create plots for the three ECG leads"""
    
    # Determine global y-limit range
    all_y = []
    for _, y in smoothed_leads:
        y_clean = np.array(y)[~np.isnan(y)]
        if len(y_clean) > 0:
            all_y.extend(y_clean)
    
    if len(all_y) > 0:
        ymin, ymax = np.min(all_y), np.max(all_y)
    else:
        ymin, ymax = -1, 1
    
    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    lead_names = ['Lead I', 'Lead II', 'Lead III']
    
    for i, (x, y) in enumerate(smoothed_leads):
        axes[i].plot(x, y, color='blue', linewidth=1.5)
        axes[i].set_title(lead_names[i], fontsize=14, fontweight='bold')
        axes[i].set_ylabel("Amplitude (pixels)", fontsize=12)
        axes[i].set_ylim([ymin, ymax])
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Horizontal Position (pixels)", fontsize=12)
    plt.tight_layout()
    
    return fig

def create_dat_file(smoothed_leads):
    """Create a .dat file with the ECG data"""
    
    output = io.StringIO()
    output.write("# Column format: TimeIndex Lead1 Lead2 Lead3\n")
    
    # Get the maximum common length
    length = min(len(lead[0]) for lead in smoothed_leads) if smoothed_leads else 0
    
    for i in range(length):
        row = [str(smoothed_leads[0][0][i])]  # Time index
        
        # Append voltage values from each lead
        for lead in smoothed_leads:
            y_val = lead[1][i]
            row.append(f"{y_val:.4f}")
        
        output.write(" ".join(row) + "\n")
    
    return output.getvalue()

def get_download_link(file_content, filename):
    """Generate a download link for the .dat file"""
    b64 = base64.b64encode(file_content.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download {filename}</a>'

# Main Streamlit App
def main():
    st.title("ECG Digitization Web App")
    st.markdown("Upload an ECG image to digitize and extract lead data with automatic processing")
    
    # Sidebar for parameters
    st.sidebar.header("Processing Parameters")
    threshold_value = st.sidebar.slider("Binary Threshold", 1, 255, 50, help="Adjust this if the ECG signal is not properly detected")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an ECG image file", 
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload a clear ECG image with 3 leads visible"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        # Process the image
        with st.spinner("Processing ECG image..."):
            try:
                # Process image
                original_gray, binary_img, processed_img = process_ecg_image(image)
                
                # Detect baselines
                segments, baselines = detect_baselines(processed_img)
                
                # Digitize leads
                digitized_leads = digitize_leads(segments, baselines, processed_img)
                
                # Apply smoothing
                smoothed_leads = smooth_leads(digitized_leads)
                
                with col2:
                    st.subheader("Processed Image")
                    st.image(processed_img, use_container_width=True, clamp=True)
                
                # Display baseline detection
                st.subheader("Baseline Detection")
                fig_baseline, ax = plt.subplots(figsize=(12, 8))
                ax.imshow(processed_img, cmap='gray')
                for i, y in enumerate(baselines):
                    ax.axhline(y=y, color='red', linestyle='--', linewidth=2, 
                             label=f'Lead {i+1} Baseline (y={y})')
                ax.set_title("Detected ECG Baselines")
                ax.axis('off')
                ax.legend()
                st.pyplot(fig_baseline)
                
                # Create and display plots
                st.subheader("Digitized ECG Leads (Savitzky-Golay Filtered)")
                fig_leads = create_plots(smoothed_leads)
                st.pyplot(fig_leads)
                
                # Create download file
                dat_content = create_dat_file(smoothed_leads)
                
                # Download button
                st.subheader("Download Results")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    # Download .dat file
                    st.download_button(
                        label="📁 Download ECG Data (.dat)",
                        data=dat_content,
                        file_name="ecg_digitized_data.dat",
                        mime="text/plain",
                        help="Download the digitized ECG data as a .dat file"
                    )
                
                with col4:
                    # Show preview of data
                    if st.button("👁️ Preview Data"):
                        st.text_area(
                            "Data Preview (first 20 lines):",
                            "\n".join(dat_content.split('\n')[:21]),
                            height=300
                        )
                
                # Display processing summary
                st.subheader("Processing Summary")
                col5, col6, col7 = st.columns(3)
                
                with col5:
                    st.metric("Detected Leads", len(smoothed_leads))
                
                with col6:
                    total_points = sum(len(lead[0]) for lead in smoothed_leads)
                    st.metric("Total Data Points", total_points)
                
                with col7:
                    if smoothed_leads:
                        duration = len(smoothed_leads[0][0])
                        st.metric("Signal Duration (pixels)", duration)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.info("Please try adjusting the threshold value or uploading a different image.")
    
    else:
        # Instructions when no file is uploaded
        st.info("^ Please upload an ECG image to begin digitization")
        
        with st.expander("ℹ️ How to use this app"):
            st.markdown("""
            1. **Upload an ECG Image**: Upload a clear ECG image with 3 visible leads
            2. **Adjust Parameters**: Use the sidebar to fine-tune processing parameters if needed
            3. **View Results**: The app will automatically process the image and show:
               - Detected baselines for each lead
               - Digitized and smoothed ECG waveforms
               - Processing summary
            4. **Download Data**: Get your digitized ECG data as a .dat file
            
            **Tips for best results:**
            - Use high-resolution, clear ECG images
            - Ensure the ECG traces are clearly visible against the background
            - The image should contain exactly 3 ECG leads arranged vertically
            """)

if __name__ == "__main__":
    main()