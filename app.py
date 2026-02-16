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
    with st.expander("📚 IS Code Reference", expanded=True):
        st.write("**IS 456:2000 Standards**\n• Max Width: 0.3mm\n• Repair: Grouting > 0.5mm")
    st.divider()
    custom_rate = st.number_input("Base Rate per mm² (Rs.)", value=10.0)
    base_visit_fee = st.number_input("Base Visiting Fee (Rs.)", value=200)
    calib = st.slider("Calibration", 0.01, 0.10, 0.05)
    sens = st.slider("Precision", 0.05, 1.0, 0.30)

# --- 3. CORE LOGIC (Length Accuracy Fix) ---
def process_analysis(img, sensitivity, calib):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, int(sensitivity * 30), 100) 
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marked_img = img.copy()
    h_data = np.zeros(img.shape[:2], dtype=np.uint8)
    max_w_px = 0.0; total_len_px = 0.0; total_area_px = 0.0; crack_pixels = []

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        (x_r, y_r), (w_r, h_r), ang = rect
        if w_r > 1.5 and h_r > 1.5:
            curr_w = min(w_r, h_r)
            # FIX: arcLength crack ki poori lambaai (perimeter) naapta hai
            # Ise 2 se divide karte hain kyunki contour line crack ke dono taraf hoti hai
            curr_l = cv2.arcLength(cnt, False) / 2 
            if curr_w > max_w_px: max_w_px = curr_w
            total_len_px += curr_l
            total_area_px += cv2.contourArea(cnt)
            cv2.drawContours(marked_img, [cnt], -1, (0, 0, 255), 2)
            cv2.drawContours(h_data, [cnt], -1, (255), -1)
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            crack_pixels.extend(gray[mask == 255])

    heatmap = cv2.applyColorMap(cv2.GaussianBlur(h_data, (51, 51), 0), cv2.COLORMAP_JET)
    mm_w = round(max_w_px * calib, 2); mm_l = round(total_len_px * calib, 2)
    mm_area = round(total_area_px * (calib ** 2), 2)
    
    if crack_pixels and mm_w > 0:
        contrast = (np.mean(gray) - np.mean(crack_pixels)) / (np.mean(gray) + 1)
        estimated_depth = round(mm_w * (0.8 + contrast), 2)
    else: estimated_depth = round(mm_w * 0.6, 2)
    estimated_depth = max(0.1, min(100, estimated_depth))

    if mm_w > 0:
        depth_n = min(estimated_depth / 50.0, 1.0)
        for cnt in contours:
            cv2.drawContours(marked_img, [cnt], -1, (int(255 * depth_n), 0, 255), int(2 + depth_n * 3))
    return marked_img, heatmap, mm_w, mm_l, mm_area, estimated_depth

def generate_stress_curve(mm_w, material, location):
    E = (35000 if material == "Concrete" else 5000) * {"Column": 1.5, "Beam": 1.3, "Slab": 1.0, "Wall": 0.8}.get(location, 0.6)
    f_strain = 0.002 + (mm_w / 60) if material == "Concrete" else 0.004 + (mm_w / 50)
    strain = np.linspace(0, f_strain * 1.5, 100)
    stress = np.maximum(E * strain * (1 - (strain / (2.2 * f_strain))), 0)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=strain, y=stress, mode='lines', name='Structural Integrity', line=dict(color='#ff5733', width=3)))
    fig.update_layout(title=f"Physics Plot: {location}", template="plotly_dark", height=280)
    return fig

def get_priority_v54(width, depth, material):
    if material == "Plaster":
        if width < 1.0 and depth < 5.0: return "LOW", "🟢", "Safe (Surface Plaster)"
        elif width < 3.0 or depth < 15.0: return "MEDIUM", "🟡", "Warning"
        return "HIGH", "🔴", "Deep Crack"
    else:
        if width < 0.3 and depth < 10.0: return "LOW", "🟢", "Safe (Hairline)"
        elif width < 0.7 or depth < 30.0: return "MEDIUM", "🟡", "Warning (Structural)"
        return "HIGH", "🔴", "CRITICAL!"

# --- 4. TABS UI ---
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
                total_repair = round((area * custom_rate * (2.5 if priority=="HIGH" else 1.5 if priority=="MEDIUM" else 1.0)) + base_visit_fee, 2)
                st.subheader(f"Result {i+1}: {f.name} ({emoji} {p_text})")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Width", f"{w} mm"); m2.metric("Length", f"{l} mm"); m3.metric("Depth", f"{depth} mm"); m4.metric("Status", p_text); m5.metric("Estimate", f"Rs. {total_repair}")
                st.image([img, m_img, h_img], caption=["Original", "Marked", "Heatmap"], width=230)
                st.plotly_chart(generate_stress_curve(w, mat, loc), use_container_width=True)
                ai_resp = client.chat.completions.create(messages=[{"role":"user","content":f"Crack {w}mm x {l}mm on {loc} ({mat}). Cause & Repair?"}], model="llama-3.1-8b-instant").choices[0].message.content
                st.info(ai_resp)
                c.execute("INSERT INTO audit_logs VALUES (?,?,?,?,?,?,?,?,?,?)", (datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%H:%M:%S'), loc, mat, f"{w}mm", f"{l}mm", f"{depth}mm", priority, f"Rs. {total_repair}", ai_resp))
                conn.commit()

with tab2:
    st.subheader(" USB External Camera")
    mat_l = st.radio("Live Material:", ["Concrete", "Plaster"], horizontal=True)
    loc_l = st.selectbox("Live Location:", ["Wall", "Column", "Beam", "Slab", "Surface"])
    webrtc_ctx = webrtc_streamer(key="usb-cam", mode=WebRtcMode.SENDRECV, rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}), media_stream_constraints={"video": True, "audio": False})
    if webrtc_ctx.video_receiver:
        if st.button("📸 Capture & Analyze Snapshot"):
            frame = webrtc_ctx.video_receiver.get_frame().to_ndarray(format="bgr24")
            m_usb, h_usb, w, l, area, depth = process_analysis(frame, sens, calib)
            st.metric("Detected Length", f"{l} mm")
            st.image([m_usb, h_usb], width=350)
    st.camera_input("Backup Camera")

with tab3:
    history = pd.read_sql_query("SELECT * FROM audit_logs ORDER BY date DESC", conn)
    st.dataframe(history, use_container_width=True)
