import streamlit as st
import cv2
import numpy as np
from groq import Groq
from fpdf import FPDF
from datetime import datetime
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import io, os, gc
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
             (date TEXT, time TEXT, location TEXT, material TEXT, width TEXT, depth TEXT, priority TEXT, cost TEXT, details TEXT)''')
conn.commit()

GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# --- 2. SIDEBAR (IS CODE REFERENCE INCLUDED) ---
with st.sidebar:
    st.header("🛡️ Engineering Panel")
    
    # IS Code Reference Section (Wapas Add Kar Diya)
    with st.expander("📚 IS Code Reference", expanded=True):
        st.write("**IS 456:2000 Standards**")
        st.write("• Max Width: 0.3mm (Durability)\n• Repair: Grouting if > 0.5mm")
        st.write("• Concrete Cover: High risk if deep")

    if st.button("🔄 Clear System Cache", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
        
    st.divider()
    custom_rate = st.number_input("Base Rate per mm² (Rs.)", value=10.0)
    base_visit_fee = st.number_input("Base Visiting Fee (Rs.)", value=200)
    calib = st.slider("Calibration (Pixels to mm)", 0.01, 0.20, 0.05)
    sens = st.slider("Precision/Sensitivity", 0.1, 1.0, 0.5)

    st.divider()
    if 'last_pdf' in st.session_state:
        st.download_button("📩 Download Last PDF Report", st.session_state.last_pdf, "Crack_Audit.pdf", "application/pdf", use_container_width=True)

# --- 3. CORE LOGIC (Accuracy Fix + No Length) ---
def process_analysis(img, sensitivity, calib, material):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Accuracy Boost: Contrast and Denoising
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    if material == "Concrete":
        # Bilateral + NlMeans for rough concrete surfaces
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    else:
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Dynamic Edge Detection
    high_t, _ = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(denoised, int(0.4 * high_t * sensitivity), int(high_t * sensitivity))
    
    kernel = np.ones((3,3), np.uint8)
    linked = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(linked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marked_img = img.copy()
    h_data = np.zeros(img.shape[:2], dtype=np.uint8)
    
    max_w_px = 0.0; total_area_px = 0.0 

    for cnt in contours:
        if cv2.arcLength(cnt, False) < 25: continue 
        area_px = cv2.contourArea(cnt)
        
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        
        # Distance Transform for Precise Width
        dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, max_val, _, _ = cv2.minMaxLoc(dt)
        curr_w_px = max_val * 2.0 
        
        if curr_w_px > max_w_px: max_w_px = curr_w_px
        total_area_px += area_px
        
        cv2.drawContours(marked_img, [cnt], -1, (0, 0, 255), 2)
        cv2.drawContours(h_data, [cnt], -1, (255), -1)

    mm_w = round(max_w_px * calib, 2)
    mm_area = round(total_area_px * (calib ** 2), 2)
    mm_d = round(mm_w * 1.8, 2) if material == "Concrete" else round(mm_w * 1.2, 2)
    
    heatmap = cv2.applyColorMap(cv2.GaussianBlur(h_data, (31, 31), 0), cv2.COLORMAP_JET)
    return marked_img, heatmap, mm_w, mm_d, mm_area

def get_priority_v54(width, material):
    if material == "Plaster":
        if width < 0.5: return "LOW", "🟢", "Safe"
        else: return "MEDIUM", "🟡", "Surface Crack"
    else:
        if width < 0.3: return "LOW", "🟢", "Safe (Hairline)"
        elif width < 1.0: return "MEDIUM", "🟡", "Warning"
        else: return "HIGH", "🔴", "CRITICAL!"

# --- 4. TABS UI ---
tab1, tab2, tab3 = st.tabs(["🚀 Audit dashboard", "📸 Live Capture", "📜 Logs"])

with tab1:
    c1, c2 = st.columns(2)
    with c1: mat = st.radio("Material:", ["Concrete", "Plaster"], horizontal=True)
    with c2: loc = st.selectbox("Location:", ["Wall", "Column", "Beam", "Slab"])
    
    f = st.file_uploader("Upload Crack Image")
    if f:
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)
        m_img, h_img, w, d, area = process_analysis(img, sens, calib, mat)
        p, emoji, p_txt = get_priority_v54(w, mat)
        cost = round((area * custom_rate) + base_visit_fee, 2)
        
        st.divider()
        st.subheader(f"Status: {emoji} {p_txt}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Width", f"{w} mm")
        col2.metric("Depth", f"{d} mm")
        col3.metric("Cost", f"Rs. {cost}")
        
        st.image([m_img, h_img], width=480)
        
        ai_resp = client.chat.completions.create(messages=[{"role":"user","content":f"{w}mm crack on {mat} {loc}. Repair advice?"}], model="llama-3.1-8b-instant").choices[0].message.content
        st.info(ai_resp)
        
        # Save to DB & Generate PDF
        c.execute("INSERT INTO audit_logs VALUES (?,?,?,?,?,?,?,?,?)", (datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%H:%M:%S'), loc, mat, f"{w}mm", f"{d}mm", p, f"Rs.{cost}", ai_resp))
        conn.commit()
        
        pdf = FPDF()
        pdf.add_page(); pdf.set_font("Arial", 'B', 16); pdf.cell(0, 10, "Crack Audit Report", 0, 1, 'C')
        pdf.set_font("Arial", size=12); pdf.ln(10); pdf.cell(0, 10, f"Result: {p_txt} | Width: {w}mm | Depth: {d}mm", 0, 1)
        st.session_state.last_pdf = pdf.output(dest='S').encode('latin-1')

with tab2:
    st.subheader("Live USB Camera Scan")
    l_mat = st.radio("Live Material:", ["Concrete", "Plaster"], horizontal=True, key="live_m")
    
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}, {"urls": ["stun:global.stun.twilio.com:3478"]}]})
    ctx = webrtc_streamer(key="live-v60", mode=WebRtcMode.SENDRECV, rtc_configuration=rtc_config, media_stream_constraints={"video": True, "audio": False})
    
    if ctx.video_receiver:
        if st.button("📸 Capture Analysis", use_container_width=True):
            frame = ctx.video_receiver.get_frame()
            img_l = frame.to_ndarray(format="bgr24")
            m, h, w, d, a = process_analysis(img_l, sens, calib, l_mat)
            p, e, t = get_priority_v54(w, l_mat)
            cost_l = round((a * custom_rate) + base_visit_fee, 2)
            
            ai_l = client.chat.completions.create(messages=[{"role":"user","content":f"{w}mm {l_mat} crack fix?"}], model="llama-3.1-8b-instant").choices[0].message.content
            st.session_state.l_res = {"m": m, "h": h, "w": w, "d": d, "p": t, "e": e, "c": cost_l, "ai": ai_l}
            
    if 'l_res' in st.session_state and st.session_state.l_res:
        r = st.session_state.l_res
        st.metric("Metrics", f"Width: {r['w']}mm | Depth: {r['d']}mm | Status: {r['p']}")
        st.image([r['m'], r['h']], width=400)
        st.info(r['ai'])

with tab3:
    st.dataframe(pd.read_sql_query("SELECT * FROM audit_logs ORDER BY date DESC", conn), use_container_width=True)
    
