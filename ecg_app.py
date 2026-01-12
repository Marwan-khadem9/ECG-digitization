import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import median_filter
from skimage.morphology import thin
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt
import io
import base64
import pandas as pd

# Set page config
st.set_page_config(
    page_title="ECG Digitization Web App",
    layout="wide"
)

def process_ecg_image(uploaded_image, threshold_value, use_dilation_thinning):
    """Process the uploaded ECG image and extract lead data"""
    
    # Step 1: Convert PIL image to grayscale
    image = uploaded_image.convert('L')
    image_np = np.array(image)
    
    # Step 2: Apply binarization
    _, binary_image = cv2.threshold(image_np, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Step 3: Signal enhancement
    # Conditional dilation
    if use_dilation_thinning:
        # Median filtering to remove noise
        denoised = median_filter(binary_image, size=3)
        
        # Invert image for morphological operations
        inverted = cv2.bitwise_not(denoised)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(inverted, kernel, iterations=1)
        
        binary_mask = (dilated // 255).astype(np.uint8)
        thinned = thin(binary_mask).astype(np.uint8) * 255
        processed_image = cv2.bitwise_not(thinned)
    else:
        processed_image = binary_image

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
        baseline_local = baselines[i] - i * (height // 3)  # baseline for this segment
        x_vals = []
        y_vals = []

        for col in range(width):
            col_data = segment[:, col].astype(float)
            black_rows = np.where(col_data > 0)[0]

            if len(black_rows) > 0:
                # Use Median instead of furthest
                r = int(np.median(black_rows))

                # --- Vertical sub-pixel refinement using 3-point parabola ---
                if 1 <= r < height - 1:
                    y_minus = col_data[r-1]
                    y0 = col_data[r]
                    y_plus = col_data[r+1]

                    denom = (y_minus - 2*y0 + y_plus)
                    delta = 0.5 * (y_minus - y_plus) / denom if abs(denom) > 1e-6 else 0

                    subpixel_row = r + delta
                else:
                    subpixel_row = r

                voltage = baseline_local - subpixel_row
                x_vals.append(col)
                y_vals.append(voltage)
            else:
                # No signal detected in this column
                x_vals.append(col)
                y_vals.append(np.nan)

        # --- Remove NaNs and interpolate horizontally to get smoother sub-pixel curve ---
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        mask = ~np.isnan(y_vals)
        x_vals_clean = x_vals[mask]
        y_vals_clean = y_vals[mask]

        # Horizontal cubic interpolation: upsample 3x
        f_interp = interp1d(x_vals_clean, y_vals_clean, kind='cubic')
        x_fine = np.linspace(x_vals_clean[0], x_vals_clean[-1], len(x_vals_clean)*3)
        y_fine = f_interp(x_fine)

        # Save digitized lead
        digitized_leads.append((x_fine, y_fine))

    return digitized_leads

def smooth_leads(digitized_leads, use_smoothing):
    """Apply Savitzky-Golay filter to smooth the leads"""
    
    smoothed_leads = []
    
    for x_vals, y_vals in digitized_leads:
        y_array = np.array(y_vals)
        
        # Handle NaNs by interpolation
        if np.isnan(y_array).any():
            not_nan = ~np.isnan(y_array)
            if np.sum(not_nan) > 1:  # Need at least 2 points for interpolation
                y_array = np.interp(np.arange(len(y_array)), np.flatnonzero(not_nan), y_array[not_nan])
        
        # Conditional Savitzky-Golay filter
        if use_smoothing:
            if len(y_array) > 9:
                window_length = min(9, len(y_array) if len(y_array) % 2 == 1 else len(y_array) - 1)
                smoothed_y = savgol_filter(y_array, window_length=window_length, polyorder=3)
            else:
                smoothed_y = y_array  # Skip filtering for very short signals
        else:
            smoothed_y = y_array  # No smoothing applied
        
        smoothed_leads.append((x_vals, smoothed_y.tolist()))
    
    return smoothed_leads

def create_plots(smoothed_leads):
    """Create plots for the three ECG leads that fit on screen"""
    
    # Determine global y-limit range
    all_y = []
    for _, y in smoothed_leads:
        y_clean = np.array(y)[~np.isnan(y)]
        if len(y_clean) > 0:
            all_y.extend(y_clean)
    all_y = np.array(all_y)/80 # Convert from pixels to mV
    
    if len(all_y) > 0:
        ymin, ymax = np.min(all_y), np.max(all_y)
    else:
        ymin, ymax = -1, 1
    
    # Create more compact plots with reduced figure height
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    lead_names = ['Lead I', 'Lead II', 'Lead III']
    
    for i, (x, y) in enumerate(smoothed_leads):
        y = np.array(y)
        axes[i].plot(x/200, y/80, color='blue', linewidth=1.5)
        axes[i].set_title(lead_names[i], fontsize=12, fontweight='bold')
        axes[i].set_ylabel("Amplitude (mV)", fontsize=10)
        axes[i].set_ylim([ymin, ymax])
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(labelsize=9)
    
    axes[-1].set_xlabel("Time (seconds)", fontsize=10)
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

# --- QUALITY METRICS FUNCTIONS ---

def get_waveform_mask_for_metrics(signal, h, w, band_height, center, shift):
    """Generate waveform mask for metric computation"""
    mask = np.zeros((h, w), dtype=np.uint8)
    signal_flipped = band_height - signal
    signal_centered = signal_flipped - np.mean(signal_flipped)
    signal_y = signal_centered + center + shift
    signal_resampled = np.interp(np.linspace(0, len(signal_y)-1, w),
                                 np.arange(len(signal_y)), signal_y).astype(int)
    for x in range(1, w):
        y1, y2 = signal_resampled[x-1], signal_resampled[x]
        if 0 <= y1 < h and 0 <= y2 < h:
            cv2.line(mask, (x-1, y1), (x, y2), 255, 1)
    return mask

def chamfer_distance(mask1, mask2):
    """Compute Chamfer distance between two masks"""
    dist_map = distance_transform_edt(1 - mask2)
    points1 = np.where(mask1 > 0)
    return np.mean(dist_map[points1]) if len(points1[0]) > 0 else np.nan

def trimmed_hausdorff(A, B, percentile=99):
    """Compute trimmed Hausdorff distance"""
    if len(A) == 0 or len(B) == 0:
        return np.nan
    dists = np.array([np.min(np.linalg.norm(B - a, axis=1)) for a in A])
    return np.percentile(dists, percentile)

def modified_hausdorff(A, B):
    """Compute modified Hausdorff distance"""
    if len(A) == 0 or len(B) == 0:
        return np.nan
    d_ab = np.mean([np.min(np.linalg.norm(B - a, axis=1)) for a in A])
    d_ba = np.mean([np.min(np.linalg.norm(A - b, axis=1)) for b in B])
    return max(d_ab, d_ba)

def hausdorff_segmentwise(pred_mask, true_mask, y_start, y_end, num_segments=5, metric_type='trimmed'):
    """Compute segment-wise Hausdorff distance"""
    pred_band = pred_mask[y_start:y_end, :]
    true_band = true_mask[y_start:y_end, :]
    segment_width = pred_band.shape[1] // num_segments
    segment_results = []

    for i in range(num_segments):
        x_start = i * segment_width
        x_end = (i+1) * segment_width if i < num_segments-1 else pred_band.shape[1]
        seg_pred = pred_band[:, x_start:x_end]
        seg_true = true_band[:, x_start:x_end]

        pred_pts = np.column_stack(np.where(seg_pred == 255))
        true_pts = np.column_stack(np.where(seg_true == 1))

        if len(pred_pts) > 0 and len(true_pts) > 0:
            if metric_type == 'trimmed':
                h1 = trimmed_hausdorff(pred_pts, true_pts, 99)
                h2 = trimmed_hausdorff(true_pts, pred_pts, 99)
                hd = max(h1, h2)
            else:  # modified
                hd = modified_hausdorff(pred_pts, true_pts)
        else:
            hd = np.nan
        segment_results.append(hd)
    return segment_results

def compute_quality_metrics(image_np, dat_content, compute_chamfer, compute_trimmed, compute_modified):
    """Compute selected quality metrics"""
    if not (compute_chamfer or compute_trimmed or compute_modified):
        return None
    
    # Preprocess original image
    _, binary = cv2.threshold(image_np, 25, 255, cv2.THRESH_BINARY)
    ecg_trace_mask = (binary == 0).astype(np.uint8)
    h, w = ecg_trace_mask.shape
    band_height = h // 3
    
    band_center = {
        "Lead1": band_height // 2,
        "Lead2": band_height + band_height // 2,
        "Lead3": 2 * band_height + band_height // 2
    }
    
    # Parse .dat file
    lines = dat_content.strip().split('\n')[1:]  # Skip header
    data = []
    for line in lines:
        values = line.strip().split()
        if len(values) >= 4:
            data.append([float(values[0]), float(values[1]), float(values[2]), float(values[3])])
    data = np.array(data)
    
    lead_indices = {"Lead1": 1, "Lead2": 2, "Lead3": 3}
    search_range = range(2, 13)
    
    results = {
        "Lead1": {},
        "Lead2": {},
        "Lead3": {}
    }
    
    for lead in ["Lead1", "Lead2", "Lead3"]:
        signal = data[:, lead_indices[lead]]
        center = band_center[lead]
        y_start = center - band_height // 2
        y_end = center + band_height // 2
        
        # Optimize for each metric
        if compute_chamfer:
            best_chamfer = float("inf")
            best_shift_chamfer = 0
            
            for shift in search_range:
                waveform_mask = get_waveform_mask_for_metrics(signal, h, w, band_height, center, shift)
                wave_band = waveform_mask[y_start:y_end, :]
                trace_band = ecg_trace_mask[y_start:y_end, :]
                chamfer_val = chamfer_distance(wave_band, trace_band)
                
                if chamfer_val < best_chamfer:
                    best_chamfer = chamfer_val
                    best_shift_chamfer = shift
            
            results[lead]["chamfer"] = best_chamfer
            results[lead]["chamfer_shift"] = best_shift_chamfer
        
        if compute_trimmed:
            best_trimmed = float("inf")
            best_shift_trimmed = 0
            
            for shift in search_range:
                waveform_mask = get_waveform_mask_for_metrics(signal, h, w, band_height, center, shift)
                segment_hd = hausdorff_segmentwise(waveform_mask, ecg_trace_mask, y_start, y_end, 5, 'trimmed')
                max_hd = np.nanmax(segment_hd)
                
                if max_hd < best_trimmed:
                    best_trimmed = max_hd
                    best_shift_trimmed = shift
            
            results[lead]["trimmed_hausdorff"] = best_trimmed
            results[lead]["trimmed_shift"] = best_shift_trimmed
        
        if compute_modified:
            best_modified = float("inf")
            best_shift_modified = 0
            
            for shift in search_range:
                waveform_mask = get_waveform_mask_for_metrics(signal, h, w, band_height, center, shift)
                segment_hd = hausdorff_segmentwise(waveform_mask, ecg_trace_mask, y_start, y_end, 5, 'modified')
                max_hd = np.nanmax(segment_hd)
                
                if max_hd < best_modified:
                    best_modified = max_hd
                    best_shift_modified = shift
            
            results[lead]["modified_hausdorff"] = best_modified
            results[lead]["modified_shift"] = best_shift_modified
    
    return results

def get_example_image_download_link(image_data, filename):
    """Create a download link for example images"""
    buffered = io.BytesIO()
    image_data.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">⬇️ Download</a>'
    return href

def show_example_images():
    """Display example ECG images that users can download and test"""
    st.subheader("📋 Example ECG Images")
    st.markdown("Download these example images to test the app:")
    
    # GitHub raw URLs for example images
    # Replace YOUR_USERNAME and YOUR_REPO with your actual GitHub details
    base_url = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/example_ecgs/"
    
    # Create example descriptions
    examples = [
        {
            "name": "Example 1: Standard ECG",
            "description": "3-lead ECG with normal sinus rhythm, clear baseline",
            "url": base_url + "example1.png",
            "threshold": "45-55"
        },
        {
            "name": "Example 2: Regular Rhythm",
            "description": "4-beat rhythm strip with prominent R waves",
            "url": base_url + "example2.png",
            "threshold": "40-50"
        },
        {
            "name": "Example 3: Variable Baseline",
            "description": "ECG with baseline drift in third lead",
            "url": base_url + "example3.png",
            "threshold": "45-60"
        },
        {
            "name": "Example 4: Bradycardia",
            "description": "Slower heart rate, 2 beats per lead",
            "url": base_url + "example4.png",
            "threshold": "40-55"
        }
    ]
    
    cols = st.columns(2)
    
    for idx in range(4):
        col = cols[idx % 2]
        
        with col:
            with st.expander(examples[idx]["name"], expanded=False):
                st.markdown(f"*{examples[idx]['description']}*")
                
                try:
                    # Display the image from GitHub
                    st.image(examples[idx]["url"], 
                             caption=f"Example ECG {idx+1}", 
                             use_container_width=True)
                    
                    st.markdown("**Download Instructions:**")
                    st.markdown("1. Right-click on the image above")
                    st.markdown("2. Select 'Save image as...'")
                    st.markdown("3. Upload it using the file uploader above")
                    
                    st.info(f"💡 Recommended threshold: {examples[idx]['threshold']}")
                    
                    # Direct download link
                    st.markdown(f"[⬇️ Direct Download Link]({examples[idx]['url']})")
                    
                except Exception as e:
                    st.warning(f"Could not load example image. [Download directly]({examples[idx]['url']})")
                


# Main Streamlit App
def main():
    st.title("ECG Digitization Web App")
    st.markdown("Upload an ECG image to digitize and extract lead data with automatic processing")
    
    # Show example images section
    with st.expander("📸 View Example ECG Images", expanded=False):
        st.markdown("""
        **Don't have an ECG image?** Try these examples to see how the app works!
        
        These sample images demonstrate different ECG patterns and are ideal for testing the digitization process.
        """)
        show_example_images()
    
    # Sidebar for parameters
    st.sidebar.header("Processing Parameters")
    threshold_value = st.sidebar.slider("Binary Threshold", 1, 255, 50, help="Adjust this if the ECG signal is not properly detected")
    
    st.sidebar.header("Processing Options")
    use_dilation_thinning = st.sidebar.checkbox("Include Dilation and Thinning", value=True, help="Apply dilation to bridge gaps in the ECG signal and morphological thinning to refine signal lines")
    use_smoothing = st.sidebar.checkbox("Include Savitzky-Golay Filter", value=True, help="Apply post-processing smoothing filter to the digitized signal")
    
    st.sidebar.header("Quality Metrics")
    st.sidebar.markdown("Select metrics to evaluate digitization quality:")
    compute_chamfer = st.sidebar.checkbox("Chamfer Distance", value=False, help="Compute Chamfer distance between digitized and original signal")
    compute_trimmed_hausdorff = st.sidebar.checkbox("Trimmed Hausdorff Distance", value=False, help="Compute 99th percentile Hausdorff distance")
    compute_modified_hausdorff = st.sidebar.checkbox("Modified Hausdorff Distance", value=False, help="Compute mean-based Hausdorff distance")
    
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
                # Process image with user settings
                original_gray, binary_img, processed_img = process_ecg_image(image, threshold_value, use_dilation_thinning)
                
                # Detect baselines
                segments, baselines = detect_baselines(processed_img)
                
                # Digitize leads
                digitized_leads = digitize_leads(segments, baselines, processed_img)
                
                # Apply smoothing with user settings
                smoothed_leads = smooth_leads(digitized_leads, use_smoothing)
                
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
                filter_status = "with" if use_smoothing else "without"
                st.subheader(f"Digitized ECG Leads ({filter_status} Savitzky-Golay Filter)")
                fig_leads = create_plots(smoothed_leads)
                st.pyplot(fig_leads)
                
                # Create download file
                dat_content = create_dat_file(smoothed_leads)
                
                # Compute quality metrics if requested
                quality_results = None
                if compute_chamfer or compute_trimmed_hausdorff or compute_modified_hausdorff:
                    with st.spinner("Computing quality metrics..."):
                        quality_results = compute_quality_metrics(
                            original_gray, 
                            dat_content, 
                            compute_chamfer, 
                            compute_trimmed_hausdorff, 
                            compute_modified_hausdorff
                        )
                
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
                        st.metric("Signal Duration (samples)", duration)
                
                # Display quality metrics if computed
                if quality_results is not None:
                    st.subheader("Quality Metrics")
                    
                    # Create metrics table
                    metrics_data = []
                    for lead in ["Lead1", "Lead2", "Lead3"]:
                        row = {"Lead": lead}
                        
                        if compute_chamfer:
                            row["Chamfer Distance"] = f"{quality_results[lead].get('chamfer', np.nan):.2f}"
                            row["Chamfer Opt. Shift"] = quality_results[lead].get('chamfer_shift', 'N/A')
                        
                        if compute_trimmed_hausdorff:
                            row["Trimmed Hausdorff"] = f"{quality_results[lead].get('trimmed_hausdorff', np.nan):.2f}"
                            row["Trimmed Opt. Shift"] = quality_results[lead].get('trimmed_shift', 'N/A')
                        
                        if compute_modified_hausdorff:
                            row["Modified Hausdorff"] = f"{quality_results[lead].get('modified_hausdorff', np.nan):.2f}"
                            row["Modified Opt. Shift"] = quality_results[lead].get('modified_shift', 'N/A')
                        
                        metrics_data.append(row)
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    st.info("ℹ️ Lower metric values indicate better digitization quality. All distances are in pixels.")
                
                # Show active processing options
                st.sidebar.markdown("---")
                st.sidebar.subheader("Active Settings")
                st.sidebar.write(f"✓ Dilation and Thinning: {'Enabled' if use_dilation_thinning else 'Disabled'}")
                st.sidebar.write(f"✓ Smoothing: {'Enabled' if use_smoothing else 'Disabled'}")
                
                if compute_chamfer or compute_trimmed_hausdorff or compute_modified_hausdorff:
                    st.sidebar.markdown("---")
                    st.sidebar.subheader("Active Metrics")
                    if compute_chamfer:
                        st.sidebar.write("✓ Chamfer Distance")
                    if compute_trimmed_hausdorff:
                        st.sidebar.write("✓ Trimmed Hausdorff")
                    if compute_modified_hausdorff:
                        st.sidebar.write("✓ Modified Hausdorff")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.info("Please try adjusting the threshold value or uploading a different image.")
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload an ECG image to begin digitization, or check out the example images above")
        
        with st.expander("ℹ️ How to use this app"):
            st.markdown("""
            1. **Upload an ECG Image**: Upload a clear ECG image with 3 visible leads
            2. **Adjust Parameters**: Use the sidebar to fine-tune processing parameters if needed
            3. **Toggle Processing Options**: Enable or disable dilation, thinning, and smoothing filters
            4. **View Results**: The app will automatically process the image and show:
               - Detected baselines for each lead
               - Digitized and smoothed ECG waveforms
               - Processing summary
            5. **Download Data**: Get your digitized ECG data as a .dat file
            
            **Tips for best results:**
            - Use high-resolution, clear ECG images
            - Ensure the ECG traces are clearly visible against the background
            - The image should contain exactly 3 ECG leads arranged vertically
            - Experiment with processing options to optimize results for your specific ECG image
            - Try the example images above if you don't have your own ECG image
            """)

if __name__ == "__main__":
    main()