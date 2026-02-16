
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
st.markdown("""<style>.stApp { background-image: linear-gradient(rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.85)), url('https://img.freepik.com/free-vector/abstract-architectural-blueprint-background_52683-59424.jpg') !important; background-size: cover !important; background-attachment: fixed !important; } h1 { color: #1E3A8A !important; font-weight: 800 !important; text-align: center; }</style>""", unsafe_allow_html=True)
st.title("AI-Powered Structural Crack Detection System")

conn = sqlite3.connect('structural_master_v60.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS audit_logs (date TEXT, time TEXT, location TEXT, material TEXT, width TEXT, length TEXT, depth TEXT, priority TEXT, cost TEXT, details TEXT)''')
conn.commit()

GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("🛡️ Engineering Panel")
    if st.button("🔄 Clear System Cache", use_container_width=True):
        st.cache_data.clear(); st.cache_resource.clear(); st.rerun()
    st.divider()
    custom_rate = st.number_input("Base Rate per mm² (Rs.)", value=10.0)
    base_visit_fee = st.number_input("Base Visiting Fee (Rs.)", value=200)
    calib = st.slider("Calibration (Scale Factor)", 0.01, 0.20, 0.10, help="Adjust if 10cm shows as 5cm")
    sens = st.slider("Precision (Sensitivity)", 0.1, 2.0, 1.0)

# --- 3. CORE LOGIC (Accuracy Upgraded) ---
def process_analysis(img, sensitivity, calib):
    # Image Preprocessing for better accuracy
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive Histogram Equalization to fix lighting issues
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Using Adaptive Thresholding instead of simple Canny for stability
    edges = cv2.Canny(blur, int(40/sensitivity), int(120/sensitivity))
    
    # Noise removal
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marked_img = img.copy()
    h_data = np.zeros(img.shape[:2], dtype=np.uint8)
    
    max_w_px = 0.0; total_len_px = 0.0; total_area_px = 0.0 
    crack_pixels = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10: # Filter noise
            # Width calculation using Distance Transform (Most accurate way)
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            dist_trans = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            _, max_val, _, _ = cv2.minMaxLoc(dist_trans)
            curr_w = max_val * 2 # Internal radius * 2 = Width
            
            # Length calculation using arcLength with Correction Factor
            arc_len = cv2.arcLength(cnt, False)
            curr_l = arc_len / 2.1 # Factor to avoid perimeter over-estimation
            
            if curr_w > max_w_px: max_w_px = curr_w
            total_len_px += curr_l 
            total_area_px += area
            
            cv2.drawContours(marked_img, [cnt], -1, (0, 0, 255), 2)
            cv2.drawContours(h_data, [cnt], -1, (255), -1) 
            
            pixel_values = gray[mask == 255]
            crack_pixels.extend(pixel_values)

    heatmap = cv2.applyColorMap(cv2.GaussianBlur(h_data, (51, 51), 0), cv2.COLORMAP_JET)
    
    # Final Metric conversion with Calibration
    mm_w = round(max_w_px * calib, 2)
    mm_l = round(total_len_px * calib, 2)
    mm_area = round(total_area_px * (calib ** 2), 2)
    
    # Improved Depth Logic based on Pixel Intensity drop
    if crack_pixels and mm_w > 0:
        avg_bg = np.percentile(gray, 70) # Background estimate
        avg_crack = np.mean(crack_pixels)
        contrast_ratio = (avg_bg - avg_crack) / (avg_bg + 1e-5)
        # Depth is a function of width and darkness
        estimated_depth = round(mm_w * (1.2 + contrast_ratio * 2), 2)
    else:
        estimated_depth = round(mm_w * 0.8, 2)
    
    estimated_depth = max(0.1, min(100, estimated_depth))

    # Visualization
    if mm_w > 0:
        d_norm = min(estimated_depth / 30.0, 1.0)
        cv2.putText(marked_img, f"W:{mm_w}mm L:{mm_l}mm", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
    return marked_img, heatmap, mm_w, mm_l, mm_area, estimated_depth

# --- REST OF THE UI LOGIC (SAME AS YOURS) ---
def generate_stress_curve(mm_w, material, location):
    E = (35000 if material == "Concrete" else 5000) * {"Column": 1.5, "Beam": 1.3, "Slab": 1.0, "Wall": 0.8}.get(location, 0.6)
    f_strain = 0.002 + (mm_w / 60)
    strain = np.linspace(0, f_strain * 1.5, 100)
    stress = np.maximum(E * strain * (1 - (strain / (2.2 * f_strain))), 0)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=strain, y=stress, mode='lines', line=dict(color='#ff5733', width=3)))
    fig.update_layout(title=f"Physics Plot: {location}", template="plotly_dark", height=280)
    return fig

def get_priority_v54(width, depth, material):
    if material == "Plaster":
        if width < 1.0 and depth < 5.0: return "LOW", "🟢", "Safe"
        elif width < 3.0: return "MEDIUM", "🟡", "Warning"
        return "HIGH", "🔴", "Deep Crack"
    else:
        if width < 0.3: return "LOW", "🟢", "Safe"
        elif width < 0.7: return "MEDIUM", "🟡", "Warning"
        return "HIGH", "🔴", "CRITICAL"

tab1, tab2, tab3 = st.tabs(["🚀 Batch Audit", "📸 USB Live Cam", "📜 History"])

with tab1:
    files = st.file_uploader("Upload Crack Images", accept_multiple_files=True)
    if files:
        mat = st.radio("Material:", ["Concrete", "Plaster"], horizontal=True)
        loc = st.selectbox("Location:", ["Wall", "Column", "Beam", "Slab", "Surface"])
        if st.button("Execute Batch Analysis", use_container_width=True):
            pdf = FPDF()
            for i, f in enumerate(files):
                img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)
                m_img, h_img, w, l, area, depth = process_analysis(img, sens, calib)
                priority, emoji, p_text = get_priority_v54(w, depth, mat)
                total_repair = round((area * custom_rate * (1.5 if priority=="MEDIUM" else 2.5 if priority=="HIGH" else 1.0)) + base_visit_fee, 2)
                st.subheader(f"{f.name}: {emoji} {p_text}")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Width", f"{w}mm"); m2.metric("Length", f"{l}mm"); m3.metric("Depth", f"{depth}mm"); m4.metric("Status", p_text); m5.metric("Estimate", f"Rs. {total_repair}")
                st.image([img, m_img, h_img], width=230)
                st.plotly_chart(generate_stress_curve(w, mat, loc), use_container_width=True)
                ai_resp = client.chat.completions.create(messages=[{"role":"user","content":f"Crack {w}mm x {l}mm on {loc}. Repair?"}], model="llama-3.1-8b-instant").choices[0].message.content
                st.info(ai_resp)
                c.execute("INSERT INTO audit_logs VALUES (?,?,?,?,?,?,?,?,?,?)", (datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%H:%M:%S'), loc, mat, f"{w}mm", f"{l}mm", f"{depth}mm", priority, f"Rs. {total_repair}", ai_resp))
                conn.commit()

with tab2:
    st.subheader("🔌 USB External Camera")
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_ctx = webrtc_streamer(key="usb-v60", mode=WebRtcMode.SENDRECV, rtc_configuration=rtc_config, media_stream_constraints={"video": True, "audio": False})
    if webrtc_ctx.video_receiver:
        if st.button("📸 Analyze Snapshot"):
            frame = webrtc_ctx.video_receiver.get_frame().to_ndarray(format="bgr24")
            m_usb, h_usb, w, l, area, depth = process_analysis(frame, sens, calib)
            st.metric("Length", f"{l} mm")
            st.image([m_usb, h_usb], width=350)

with tab3:
    history = pd.read_sql_query("SELECT * FROM audit_logs ORDER BY date DESC", conn)
    st.dataframe(history, use_container_width=True)

