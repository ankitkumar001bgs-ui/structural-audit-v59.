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

# --- 1. SETUP & THEME (v58 Base) ---
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

st.title("🏗️ Structural Audit AI Portal v59")

# Database Setup
conn = sqlite3.connect('structural_master_v59.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS audit_logs 
             (date TEXT, time TEXT, location TEXT, material TEXT, width TEXT, depth TEXT, priority TEXT, cost TEXT, details TEXT)''')
conn.commit()

GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# --- 2. SIDEBAR (v58 Same) ---
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

# --- 3. CORE LOGIC (With Visual 3D Depth) ---
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
        x, y, wb, hb = cv2.boundingRect(cnt)
        if wb > 1.5 or hb > 1.5:
            w_px = min(wb, hb); l_px = max(wb, hb)
            if w_px > max_w_px: max_w_px = w_px
            total_len_px += l_px; total_area_px += cv2.contourArea(cnt)
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
    
    # 3D Depth Estimation logic (from v57/v58)
    estimated_depth = 0.0
    if crack_pixels and mm_w > 0:
        avg_crack = np.mean(crack_pixels)
        avg_bg = np.mean(gray)
        contrast = (avg_bg - avg_crack) / (avg_bg + 1)
        estimated_depth = round(mm_w * (0.8 + contrast), 2)
    else:
        estimated_depth = round(mm_w * 0.6, 2)
    
    # Clamp depth to a reasonable range
    if estimated_depth < 0.1: estimated_depth = 0.1
    if estimated_depth > 100: estimated_depth = 100 

    # --- Visual 3D Depth Effect on Marked Image ---
    # Intensify blue color based on depth, simulate shadow
    if mm_w > 0 and estimated_depth > 0:
        depth_norm = min(estimated_depth / 50.0, 1.0) # Normalize depth for intensity (max 50mm for effect)
        # Create a darker, thicker blue line for deeper cracks
        for cnt in contours:
            cv2.drawContours(marked_img, [cnt], -1, (int(255 * depth_norm), 0, 255), int(2 + depth_norm * 3)) # Bolder blue
            # Add a slight shadow effect (darker outline)
            cv2.drawContours(marked_img, [cnt], -1, (int(100 * depth_norm), 0, int(100 * depth_norm)), int(1 + depth_norm * 1), lineType=cv2.LINE_AA)
            
    return marked_img, heatmap, mm_w, mm_l, mm_area, estimated_depth

def generate_stress_curve(mm_w, material, location):
    E_base = 35000 if material == "Concrete" else 5000 
    loc_factor = {"Column": 1.5, "Beam": 1.3, "Slab": 1.0, "Wall": 0.8}.get(location, 0.6) # Enhanced loc_factor
    E = E_base * loc_factor
    
    # Failure strain depends on width and material (Concrete fails earlier for same strain)
    if material == "Concrete":
        failure_strain = 0.002 + (mm_w / 60) 
    else: # Plaster can deform more before 'failure' in terms of structural load
        failure_strain = 0.004 + (mm_w / 50) 
        
    strain = np.linspace(0, failure_strain * 1.5, 100)
    
    # Stress-Strain Parabolic Curve (IS 456 for concrete, adjusted for plaster)
    stress = np.maximum(E * strain * (1 - (strain / (2.2 * failure_strain))), 0) # Adjusted constant for better curve shape
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strain, y=stress, mode='lines', name='Structural Integrity', line=dict(color='#ff5733', width=3)))
    
    # Calculate failure point (max stress)
    max_stress_idx = np.argmax(stress)
    fig.add_trace(go.Scatter(x=[strain[max_stress_idx]], y=[stress[max_stress_idx]], 
                             mode='markers', name='Failure Point', 
                             marker=dict(color='red', size=10, symbol='x', line=dict(width=2, color='DarkRed')))) # Clearer X mark
    
    fig.update_layout(title=f"Physics Plot: {location} ({material})", 
                      xaxis_title="Strain (ε)", yaxis_title="Stress (MPa)",
                      template="plotly_dark", height=280)
    return fig

def get_priority_v54(width, depth, material):
    # Priority logic now more sensitive to depth
    if material == "Plaster":
        if width < 1.0 and depth < 5.0: return "LOW", "🟢", "Safe (Surface Plaster Crack)"
        elif (1.0 <= width < 3.0 and depth < 10.0) or (depth >= 5.0 and depth < 15.0): return "MEDIUM", "🟡", "Warning"
        else: return "HIGH", "🔴", "Deep Crack (Internal Check Req.)"
    else: # Concrete
        if width < 0.3 and depth < 10.0: return "LOW", "🟢", "Safe (Hairline)"
        elif (0.3 <= width < 0.7 and depth < 20.0) or (depth >= 10.0 and depth < 30.0): return "MEDIUM", "🟡", "Warning (Structural)"
        else: return "HIGH", "🔴", "CRITICAL (Immediate Action!)"

# --- 4. TABS UI ---
tab1, tab2, tab3 = st.tabs(["🚀 Batch Audit", "📸 Live Cam", "📜 History"])

with tab1:
    files = st.file_uploader("Upload Crack Images", accept_multiple_files=True, key="file_uploader_v59")
    if files:
        c1, c2 = st.columns(2)
        with c1: mat = st.radio("Material:", ["Concrete", "Plaster"], horizontal=True)
        with c2: loc = st.selectbox("Location:", ["Wall", "Column", "Beam", "Slab", "Surface"])
            
        if st.button("Execute Batch Analysis", use_container_width=True):
            gc.collect()
            pdf = FPDF()
            pdf.set_font("Arial", size=10) # Set default font
            
            for i, f in enumerate(files):
                img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)
                m_img, h_img, w, l, area, depth = process_analysis(img, sens, calib)
                priority, emoji, p_text = get_priority_v54(w, depth, mat)
                
                total_repair = round((area * custom_rate * (1.5 if priority=="MEDIUM" else 2.5 if priority=="HIGH" else 1.0)) + base_visit_fee, 2)
                
                # UI Preview
                st.subheader(f"Result {i+1}: {f.name} ({emoji} {p_text})")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Width", f"{w} mm")
                m2.metric("Depth", f"{depth} mm")
                m3.metric("Status", p_text)
                m4.metric("Estimate", f"Rs. {total_repair}")
                
                # Image Layout Fix: Use columns to display images side-by-side
                img_col1, img_col2, img_col3 = st.columns(3)
                with img_col1: st.image(img, caption="Original", use_column_width=True)
                with img_col2: st.image(m_img, caption="Marked (3D Depth)", use_column_width=True) # Updated caption
                with img_col3: st.image(h_img, caption="Thermal Heatmap", use_column_width=True)
                
                graph_key = f"graph_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{i}"
                fig = generate_stress_curve(w, mat, loc)
                st.plotly_chart(fig, use_container_width=True, key=graph_key)
                
                prompt = f"Analyze {w}mm width and {depth}mm depth crack on {loc} ({mat}). Priority: {priority}. Give detailed Cause & Repair solutions."
                ai_resp = client.chat.completions.create(messages=[{"role":"user","content":prompt}], model="llama-3.1-8b-instant").choices[0].message.content
                st.info(ai_resp)
                
                # Database (v58 Same)
                c.execute("INSERT INTO audit_logs VALUES (?,?,?,?,?,?,?,?,?)", 
                         (datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%H:%M:%S'), loc, mat, f"{w}mm", f"{depth}mm", priority, f"Rs. {total_repair}", ai_resp))
                conn.commit()
            
                # PDF Export (v58 Same)
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, f"STRUCTURAL REPORT - {f.name}", 1, 1, 'C')
                pdf.set_font("Arial", size=10)
                pdf.cell(0, 8, f"Location: {loc} | Material: {mat} | Width: {w}mm | Depth: {depth}mm", ln=1)
                pdf.cell(0, 8, f"Priority: {priority} | Estimate: Rs. {total_repair}", ln=1)
                
                o_p, m_p, h_p, g_p = f"o_{i}.jpg", f"m_{i}.jpg", f"h_{i}.jpg", f"g_{i}.png"
                cv2.imwrite(o_p, img); cv2.imwrite(m_p, m_img); cv2.imwrite(h_p, h_img); fig.write_image(g_p)
                
                pdf.image(o_p, 10, 40, 90, 60); pdf.image(m_p, 105, 40, 90, 60)
                pdf.image(h_p, 10, 105, 90, 60); pdf.image(g_p, 105, 105, 90, 60)
                
                pdf.add_page()
                pdf.set_font("Arial", size=10)
                pdf.multi_cell(0, 6, txt=ai_resp.encode('latin-1', 'ignore').decode('latin-1'))
                
            st.download_button("📥 Download PDF Report", bytes(pdf.output(dest='S')), "Audit_Report.pdf", use_container_width=True)

with tab2: # Live Cam (v58 with depth info)
    live = st.camera_input("Scan Crack")
    if live:
        img_l = cv2.imdecode(np.frombuffer(live.read(), np.uint8), 1)
        m_l, _, w_l, _, _, d_l = process_analysis(img_l, sens, calib)
        st.image(m_l, caption=f"Detected: {w_l}mm | Depth: {d_l}mm", use_column_width=True)

with tab3: # History (v58 Same)
    history = pd.read_sql_query("SELECT * FROM audit_logs ORDER BY date DESC", conn)
    st.dataframe(history, use_container_width=True)

