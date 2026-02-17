import streamlit as st
import cv2
import numpy as np
from groq import Groq
from fpdf import FPDF
from datetime import datetime
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import io
import os
import gc
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- 1. SETUP & THEME ---
st.set_page_config(page_title="Structural AI Audit Pro v60", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.85)), 
                          url('https://img.freepik.com/free-vector/abstract-architectural-blueprint-background_52683-59424.jpg') !important;
        background-size: cover !important;
        background-attachment: fixed !important;
    }
    h1 { color: #1E3A8A !important; font-weight: 800 !important; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.title("AI-Powered Structural Crack Detection System")

# Database Setup
conn = sqlite3.connect('structural_master_v60.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS audit_logs 
             (date TEXT, time TEXT, location TEXT, material TEXT, width TEXT, length TEXT, depth TEXT, priority TEXT, cost TEXT, details TEXT)''')
conn.commit()

GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("🛡️ Engineering Panel")
    if st.button("🔄 Clear System Cache", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
        
    with st.expander("📚 IS Code Reference", expanded=True):
        st.write("**IS 456:2000 Standards**")
        st.write("• Max Width: 0.3mm\n• Repair: Grouting > 0.5mm")
    st.divider()
    custom_rate = st.number_input("Base Rate per mm² (Rs.)", value=10.0)
    base_visit_fee = st.number_input("Base Visiting Fee (Rs.)", value=200)
    calib = st.slider("Calibration (Pixels to mm)", 0.01, 0.20, 0.05) # Increased range for bigger cracks
    sens = st.slider("Precision", 0.05, 1.0, 0.30)

# --- 3. CORE LOGIC (Accuracy Fix for Concrete) ---
def process_analysis(img, sensitivity, calib):
    # Image Pre-processing for Concrete Texture
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Bilateral Filter: Noise hatata hai lekin edges (crack) ko sharp rakhta hai
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    blur = cv2.GaussianBlur(denoised, (5, 5), 0)
    
    # Adaptive Canny
    edges = cv2.Canny(blur, int(sensitivity * 30), 100) 
    
    # Morphological Closing: Concrete ke toote edges ko jodne ke liye
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marked_img = img.copy()
    h_data = np.zeros(img.shape[:2], dtype=np.uint8)
    
    max_w_px = 0.0; total_len_px = 0.0; total_area_px = 0.0 
    crack_pixels = []

    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < 50: continue # Noise filter

        # Distance Transform Logic for Accurate Width
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, max_val, _, _ = cv2.minMaxLoc(dist_transform)
        
        # Current Width = 2 * Distance from center to edge
        curr_w_px = max_val * 2.0 
        
        # Length calculation (Arc length / 2)
        curr_l_px = cv2.arcLength(cnt, True) / 2.0
        
        if curr_w_px > max_w_px: max_w_px = curr_w_px
        total_len_px += curr_l_px
        total_area_px += area_px
        
        # Visual Markers
        cv2.drawContours(marked_img, [cnt], -1, (0, 0, 255), 2)
        cv2.drawContours(h_data, [cnt], -1, (255), -1)
        
        pixel_vals = gray[mask == 255]
        crack_pixels.extend(pixel_vals)

    # Pixel to MM Conversion
    mm_w = round(max_w_px * calib, 2)
    mm_l = round(total_len_px * calib, 2)
    mm_area = round(total_area_px * (calib ** 2), 2)
    
    heatmap = cv2.applyColorMap(cv2.GaussianBlur(h_data, (51, 51), 0), cv2.COLORMAP_JET)
    
    # Depth logic with Concrete/Plaster contrast adjustment
    if crack_pixels and mm_w > 0:
        avg_crack = np.mean(crack_pixels)
        avg_bg = np.mean(gray)
        contrast = (avg_bg - avg_crack) / (avg_bg + 1)
        estimated_depth = round(mm_w * (1.1 + contrast), 2) 
    else:
        estimated_depth = round(mm_w * 0.7, 2)
    
    if estimated_depth < 0.1: estimated_depth = 0.1
    if estimated_depth > 100: estimated_depth = 100 

    # 3D Depth Visual Effect
    if mm_w > 0:
        depth_norm = min(estimated_depth / 50.0, 1.0)
        for cnt in contours:
            cv2.drawContours(marked_img, [cnt], -1, (int(255 * depth_norm), 0, 255), int(2 + depth_norm * 3))
            
    return marked_img, heatmap, mm_w, mm_l, mm_area, estimated_depth

def generate_stress_curve(mm_w, material, location):
    E_base = 35000 if material == "Concrete" else 5000 
    loc_factor = {"Column": 1.5, "Beam": 1.3, "Slab": 1.0, "Wall": 0.8}.get(location, 0.6)
    E = E_base * loc_factor
    failure_strain = 0.002 + (mm_w / 60) if material == "Concrete" else 0.004 + (mm_w / 50)
    strain = np.linspace(0, failure_strain * 1.5, 100)
    stress = np.maximum(E * strain * (1 - (strain / (2.2 * failure_strain))), 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strain, y=stress, mode='lines', name='Structural Integrity', line=dict(color='#ff5733', width=3)))
    max_stress_idx = np.argmax(stress)
    fig.add_trace(go.Scatter(x=[strain[max_stress_idx]], y=[stress[max_stress_idx]], 
                             mode='markers', name='Failure Point', 
                             marker=dict(color='red', size=10, symbol='x', line=dict(width=2, color='DarkRed'))))
    fig.update_layout(title=f"Physics Plot: {location}", template="plotly_dark", height=280)
    return fig

def get_priority_v54(width, depth, material):
    if material == "Plaster":
        if width < 1.0 and depth < 5.0: return "LOW", "🟢", "Safe (Surface Plaster)"
        elif (1.0 <= width < 3.0 and depth < 10.0) or (depth >= 5.0 and depth < 15.0): return "MEDIUM", "🟡", "Warning"
        else: return "HIGH", "🔴", "Deep Crack"
    else:
        if width < 0.3 and depth < 10.0: return "LOW", "🟢", "Safe (Hairline)"
        elif (0.3 <= width < 0.7 and depth < 20.0) or (depth >= 10.0 and depth < 30.0): return "MEDIUM", "🟡", "Warning (Structural)"
        else: return "HIGH", "🔴", "CRITICAL!"

# --- 4. TABS UI ---
tab1, tab2, tab3 = st.tabs(["🚀 Batch Audit", "📸 USB Live Cam", "📜 History"])

with tab1:
    files = st.file_uploader("Upload Crack Images", accept_multiple_files=True, key="file_uploader_v60")
    if files:
        c1, c2 = st.columns(2)
        with c1: mat = st.radio("Material:", ["Concrete", "Plaster"], horizontal=True)
        with c2: loc = st.selectbox("Location:", ["Wall", "Column", "Beam", "Slab", "Surface"])
            
        if st.button("Execute Batch Analysis", use_container_width=True):
            gc.collect()
            pdf = FPDF()
            
            for i, f in enumerate(files):
                img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)
                m_img, h_img, w, l, area, depth = process_analysis(img, sens, calib)
                priority, emoji, p_text = get_priority_v54(w, depth, mat)
                total_repair = round((area * custom_rate * (1.5 if priority=="MEDIUM" else 2.5 if priority=="HIGH" else 1.0)) + base_visit_fee, 2)
                
                st.subheader(f"Result {i+1}: {f.name} ({emoji} {p_text})")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Width", f"{w} mm")
                m2.metric("Length", f"{l} mm") 
                m3.metric("Depth", f"{depth} mm")
                m4.metric("Status", p_text)
                m5.metric("Estimate", f"Rs. {total_repair}")
                
                img_col1, img_col2, img_col3 = st.columns(3)
                with img_col1: st.image(img, caption="Original", use_column_width=True)
                with img_col2: st.image(m_img, caption="Marked (3D Accuracy)", use_column_width=True)
                with img_col3: st.image(h_img, caption="Heatmap", use_column_width=True)
                
                fig = generate_stress_curve(w, mat, loc)
                st.plotly_chart(fig, use_container_width=True, key=f"g_{i}")
                
                prompt = f"Analyze {w}mm width, {l}mm length and {depth}mm depth crack on {loc} ({mat}). Priority: {priority}. Cause & Repair?"
                ai_resp = client.chat.completions.create(messages=[{"role":"user","content":prompt}], model="llama-3.1-8b-instant").choices[0].message.content
                st.info(ai_resp)
                
                c.execute("INSERT INTO audit_logs VALUES (?,?,?,?,?,?,?,?,?,?)", 
                         (datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%H:%M:%S'), loc, mat, f"{w}mm", f"{l}mm", f"{depth}mm", priority, f"Rs. {total_repair}", ai_resp))
                conn.commit()
            
                # PDF LOGIC
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, "STRUCTURAL CRACK REPORT", 0, 1, 'C')
                pdf.set_font("Arial", size=12)
                pdf.cell(0, 8, f"Report for: {f.name}", 0, 1, 'L')
                pdf.cell(0, 8, f"Width: {w}mm | Length: {l}mm | Depth: {depth}mm", 0, 1, 'L')
                pdf.ln(5)

                orig_path = f"temp_orig_{i}.png"; mark_path = f"temp_mark_{i}.png"; heat_path = f"temp_heat_{i}.png"
                cv2.imwrite(orig_path, img); cv2.imwrite(mark_path, m_img); cv2.imwrite(heat_path, h_img)
                
                pdf.image(orig_path, x=10, y=pdf.get_y(), w=60)
                pdf.image(mark_path, x=75, y=pdf.get_y(), w=60)
                pdf.image(heat_path, x=140, y=pdf.get_y(), w=60)
                pdf.ln(70) 

                pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "AI Recommendations:", 0, 1, 'L')
                pdf.set_font("Arial", size=10)
                clean_text = ai_resp.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 5, txt=clean_text)
                
                os.remove(orig_path); os.remove(mark_path); os.remove(heat_path)
            
            try:
                pdf_output = pdf.output(dest='S')
                pdf_bytes = pdf_output.encode('latin-1') if isinstance(pdf_output, str) else pdf_output
                st.download_button(label="📥 Download PDF Report", data=pdf_bytes, file_name="Audit_Report.pdf", mime="application/pdf", use_container_width=True)
            except Exception as e:
                st.error(f"PDF Error: {str(e)}")

with tab2:
    st.subheader("🔌 USB External Camera")
    c_live1, c_live2 = st.columns(2)
    with c_live1: mat_l = st.radio("Live Material:", ["Concrete", "Plaster"], horizontal=True, key="mat_live")
    with c_live2: loc_l = st.selectbox("Live Location:", ["Wall", "Column", "Beam", "Slab", "Surface"], key="loc_live")

    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_ctx = webrtc_streamer(key="usb-cam-v60", mode=WebRtcMode.SENDRECV, rtc_configuration=rtc_config, media_stream_constraints={"video": True, "audio": False})
    
    if webrtc_ctx.video_receiver:
        if st.button("📸 Capture & Analyze Snapshot", use_container_width=True):
            try:
                frame = webrtc_ctx.video_receiver.get_frame()
                img_usb = frame.to_ndarray(format="bgr24")
                m_usb, h_usb, w, l, area, depth = process_analysis(img_usb, sens, calib)
                priority, emoji, p_text = get_priority_v54(w, depth, mat_l)
                total_repair = round((area * custom_rate * (1.5 if priority=="MEDIUM" else 2.5 if priority=="HIGH" else 1.0)) + base_visit_fee, 2)
                
                st.divider()
                st.subheader(f"Live Result: {emoji} {p_text}")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Width", f"{w} mm"); m2.metric("Length", f"{l} mm"); m3.metric("Depth", f"{depth} mm"); m4.metric("Status", p_text); m5.metric("Estimate", f"Rs. {total_repair}")
                st.image([m_usb, h_usb], caption=["Marked Detection", "Heatmap"], width=350)
                
                fig_l = generate_stress_curve(w, mat_l, loc_l)
                st.plotly_chart(fig_l, use_container_width=True)
                
                prompt_l = f"Analyze {w}mm width, {l}mm length and {depth}mm depth crack on {loc_l} ({mat_l}). Priority: {priority}. Cause & Repair?"
                ai_resp_l = client.chat.completions.create(messages=[{"role":"user","content":prompt_l}], model="llama-3.1-8b-instant").choices[0].message.content
                st.info(ai_resp_l)
                
                c.execute("INSERT INTO audit_logs VALUES (?,?,?,?,?,?,?,?,?,?)", 
                         (datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%H:%M:%S'), loc_l, mat_l, f"{w}mm", f"{l}mm", f"{depth}mm", priority, f"Rs. {total_repair}", ai_resp_l))
                conn.commit()
            except Exception as e:
                st.error(f"Live Error: {e}")
    
    st.divider()
    live = st.camera_input("Default Camera (Mobile/Front)")
    if live:
        img_l = cv2.imdecode(np.frombuffer(live.read(), np.uint8), 1)
        m_l, _, w_l, l_l, _, d_l = process_analysis(img_l, sens, calib)
        st.image(m_l, caption=f"Detected: W:{w_l}mm | L:{l_l}mm | D:{d_l}mm", use_column_width=True)

with tab3:
    history = pd.read_sql_query("SELECT * FROM audit_logs ORDER BY date DESC", conn)
    st.dataframe(history, use_container_width=True)


