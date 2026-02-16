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
# Nayi libraries for USB Camera
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- 1. SETUP & THEME (v59 Final) ---
st.set_page_config(page_title="Structural AI Audit Pro v59", layout="wide")

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

# Database Setup (Added 'length' column)
conn = sqlite3.connect('structural_master_v59.db', check_same_thread=False)
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
    calib = st.slider("Calibration", 0.01, 0.10, 0.05)
    sens = st.slider("Precision", 0.05, 1.0, 0.30)

# --- 3. CORE LOGIC (With Length & Improved Precision) ---
def process_analysis(img, sensitivity, calib):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, int(sensitivity * 30), 100) 
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marked_img = img.copy()
    h_data = np.zeros(img.shape[:2], dtype=np.uint8)
    
    max_w_px = 0.0; total_len_px = 0.0; total_area_px = 0.0 
    crack_pixels = []

    for cnt in contours:
        # Precision improvement using minAreaRect
        rect = cv2.minAreaRect(cnt)
        (x, y), (w_rect, h_rect), angle = rect
        if w_rect > 1.5 or h_rect > 1.5:
            curr_w = min(w_rect, h_rect)
            curr_l = max(w_rect, h_rect)
            
            if curr_w > max_w_px: max_w_px = curr_w
            total_len_px += curr_l
            total_area_px += cv2.contourArea(cnt)
            
            # Drawing 3D effect contours
            cv2.drawContours(marked_img, [cnt], -1, (0, 0, 255), 2)
            cv2.drawContours(h_data, [cnt], -1, (255), -1)
            
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            pixel_values = gray[mask == 255]
            crack_pixels.extend(pixel_values)

    heatmap = cv2.applyColorMap(cv2.GaussianBlur(h_data, (51, 51), 0), cv2.COLORMAP_JET)
    mm_w = round(max_w_px * calib, 2)
    mm_l = round(total_len_px * calib, 2)
    mm_area = round(total_area_px * (calib ** 2), 2)
    
    # Depth calculation based on contrast
    avg_bg = np.mean(gray)
    avg_crack = np.mean(crack_pixels) if crack_pixels else avg_bg
    contrast = (avg_bg - avg_crack) / (avg_bg + 1)
    estimated_depth = round(mm_w * (0.8 + contrast * 2), 2)
    estimated_depth = max(0.1, min(100.0, estimated_depth))
            
    return marked_img, heatmap, mm_w, mm_l, mm_area, estimated_depth

def generate_stress_curve(mm_w, material, location):
    E_base = 35000 if material == "Concrete" else 5000 
    loc_factor = {"Column": 1.5, "Beam": 1.3, "Slab": 1.0, "Wall": 0.8}.get(location, 0.6)
    E = E_base * loc_factor
    failure_strain = 0.002 + (mm_w / 60)
    strain = np.linspace(0, failure_strain * 1.5, 100)
    stress = np.maximum(E * strain * (1 - (strain / (2.2 * failure_strain))), 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strain, y=stress, mode='lines', name='Structural Integrity', line=dict(color='#ff5733', width=3)))
    fig.update_layout(title=f"Physics Plot: {location}", template="plotly_dark", height=280)
    return fig

def get_priority_v54(width, depth, material):
    if material == "Plaster":
        if width < 1.0 and depth < 5.0: return "LOW", "🟢", "Safe (Surface)"
        else: return "HIGH", "🔴", "Deep Crack"
    else:
        if width < 0.3: return "LOW", "🟢", "Safe"
        elif width < 1.0: return "MEDIUM", "🟡", "Warning"
        else: return "HIGH", "🔴", "CRITICAL!"

# --- 4. TABS UI ---
tab1, tab2, tab3 = st.tabs(["🚀 Batch Audit", "📸 USB Live Cam", "📜 History"])

with tab1:
    files = st.file_uploader("Upload Crack Images", accept_multiple_files=True, key="file_uploader_v59")
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
                total_repair = round((area * custom_rate) + base_visit_fee, 2)
                
                st.subheader(f"Result {i+1}: {f.name} ({emoji} {p_text})")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Width", f"{w}mm"); m2.metric("Length", f"{l}mm"); m3.metric("Depth", f"{depth}mm"); m4.metric("Status", p_text); m5.metric("Estimate", f"Rs.{total_repair}")
                
                st.image([img, m_img, h_img], width=300)
                st.plotly_chart(generate_stress_curve(w, mat, loc), use_container_width=True)
                
                prompt = f"Crack {w}mm x {l}mm on {loc}. Suggest repair."
                ai_resp = client.chat.completions.create(messages=[{"role":"user","content":prompt}], model="llama-3.1-8b-instant").choices[0].message.content
                st.info(ai_resp)
                
                c.execute("INSERT INTO audit_logs VALUES (?,?,?,?,?,?,?,?,?,?)", 
                         (datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%H:%M:%S'), loc, mat, f"{w}mm", f"{l}mm", f"{depth}mm", priority, f"Rs. {total_repair}", ai_resp))
                conn.commit()

                # Original PDF Logic (Restored)
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, f"STRUCTURAL REPORT - {f.name}", 1, 1, 'C')
                pdf.set_font("Arial", size=10)
                pdf.cell(0, 8, f"Location: {loc} | Width: {w}mm | Length: {l}mm | Depth: {depth}mm", ln=1)
                clean_text = ai_resp.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 6, txt=clean_text)

            pdf_out = pdf.output(dest='S')
            st.download_button("📥 Download PDF Report", data=pdf_out.encode('latin-1'), file_name="Report.pdf")

with tab2:
    st.subheader("🔌 USB External Camera")
    cl1, cl2 = st.columns(2)
    with cl1: l_mat = st.radio("Live Material:", ["Concrete", "Plaster"], horizontal=True, key="l_mat")
    with cl2: l_loc = st.selectbox("Live Location:", ["Wall", "Column", "Beam", "Slab", "Surface"], key="l_loc")

    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_ctx = webrtc_streamer(key="usb-cam", mode=WebRtcMode.SENDRECV, rtc_configuration=rtc_config, media_stream_constraints={"video": True, "audio": False})
    
    if webrtc_ctx.video_receiver:
        if st.button("📸 Capture & Full Analysis", use_container_width=True):
            frame = webrtc_ctx.video_receiver.get_frame()
            img_usb = frame.to_ndarray(format="bgr24")
            m, h, w, l, area, d = process_analysis(img_usb, sens, calib)
            priority, emoji, p_text = get_priority_v54(w, d, l_mat)
            cost = round((area * custom_rate) + base_visit_fee, 2)
            
            # Yahan ab poora details show hoga
            st.divider()
            st.subheader(f"LIVE ANALYSIS: {emoji} {p_text}")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Width", f"{w}mm"); col2.metric("Length", f"{l}mm"); col3.metric("Depth", f"{d}mm"); col4.metric("Status", p_text); col5.metric("Cost", f"Rs.{cost}")
            
            st.image([m, h], width=400)
            st.plotly_chart(generate_stress_curve(w, l_mat, l_loc), use_container_width=True)
            
            prompt_l = f"Analyze {w}mm x {l}mm crack on {l_loc}. Repair?"
            ai_resp_l = client.chat.completions.create(messages=[{"role":"user","content":prompt_l}], model="llama-3.1-8b-instant").choices[0].message.content
            st.info(f"🤖 AI: {ai_resp_l}")
            
            c.execute("INSERT INTO audit_logs VALUES (?,?,?,?,?,?,?,?,?,?)", (datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%H:%M:%S'), l_loc, l_mat, f"{w}mm", f"{l}mm", f"{d}mm", priority, f"Rs. {cost}", ai_resp_l))
            conn.commit()

with tab3:
    history = pd.read_sql_query("SELECT * FROM audit_logs ORDER BY date DESC", conn)
    st.dataframe(history, use_container_width=True)

