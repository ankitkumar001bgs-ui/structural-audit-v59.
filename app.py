import streamlit as st
import cv2
import numpy as np
import pandas as pd
import sqlite3
import os
from datetime import datetime
from skimage.morphology import skeletonize
from ultralytics import YOLO
from fpdf import FPDF
from PIL import Image

# ============================
# CONFIGURATION
# ============================

CALIBRATION = 0.05  # mm per pixel
DB = "crack_ai_pro.db"

MODEL_PATH = "crack_model.pt"

# ============================
# LOAD YOLO MODEL
# ============================

@st.cache_resource
def load_model():

    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)

    return None

model = load_model()

# ============================
# DATABASE
# ============================

def init_db():

    conn = sqlite3.connect(DB)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS logs(

    date TEXT,
    time TEXT,
    width REAL,
    length REAL,
    area REAL,
    severity TEXT

    )
    """)

    conn.commit()
    conn.close()

init_db()

# ============================
# SEVERITY CLASSIFIER
# ============================

def classify(width):

    if width < 0.3:
        return "LOW"

    elif width < 0.7:
        return "MEDIUM"

    else:
        return "HIGH"

# ============================
# OPENCV FALLBACK DETECTOR
# ============================

def opencv_detect(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3)

    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        2
    )

    kernel = np.ones((3,3),np.uint8)

    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    skeleton = skeletonize(clean//255)

    contours,_ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_width = 0
    total_area = 0

    marked = image.copy()

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 20:
            continue

        total_area += area

        rect = cv2.minAreaRect(cnt)

        w,h = rect[1]

        width = min(w,h)

        max_width = max(max_width,width)

        cv2.drawContours(marked,[cnt],-1,(0,0,255),2)

    length = np.sum(skeleton)

    return max_width,length,total_area,marked

# ============================
# YOLO DETECTOR
# ============================

def yolo_detect(image):

    results = model.predict(image, conf=0.25)

    marked = image.copy()

    max_width=0
    total_length=0
    total_area=0

    if results[0].masks is None:

        return None

    masks = results[0].masks.data.cpu().numpy()

    for mask in masks:

        binary = (mask*255).astype(np.uint8)

        skeleton = skeletonize(binary//255)

        total_length += np.sum(skeleton)

        contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:

            area=cv2.contourArea(cnt)

            total_area+=area

            rect=cv2.minAreaRect(cnt)

            w,h=rect[1]

            width=min(w,h)

            max_width=max(max_width,width)

            cv2.drawContours(marked,[cnt],-1,(0,0,255),2)

    return max_width,total_length,total_area,marked

# ============================
# MAIN DETECTOR
# ============================

def detect(image):

    if model:

        result = yolo_detect(image)

        if result:

            w,l,a,marked=result

        else:

            w,l,a,marked=opencv_detect(image)

    else:

        w,l,a,marked=opencv_detect(image)

    width_mm=w*CALIBRATION

    length_mm=l*CALIBRATION

    area_mm=a*(CALIBRATION**2)

    severity=classify(width_mm)

    save_log(width_mm,length_mm,area_mm,severity)

    return width_mm,length_mm,area_mm,severity,marked

# ============================
# SAVE LOG
# ============================

def save_log(w,l,a,s):

    conn=sqlite3.connect(DB)

    conn.execute("INSERT INTO logs VALUES(?,?,?,?,?,?)",

    (
    datetime.now().date(),
    datetime.now().time(),
    w,l,a,s
    ))

    conn.commit()
    conn.close()

# ============================
# PDF REPORT
# ============================

def generate_pdf(width,length,area,severity):

    pdf=FPDF()

    pdf.add_page()

    pdf.set_font("Arial",size=16)

    pdf.cell(200,10,"CRACK ANALYSIS REPORT",ln=True)

    pdf.set_font("Arial",size=12)

    pdf.cell(200,10,f"Width: {width:.2f} mm",ln=True)
    pdf.cell(200,10,f"Length: {length:.2f} mm",ln=True)
    pdf.cell(200,10,f"Area: {area:.2f} mm2",ln=True)
    pdf.cell(200,10,f"Severity: {severity}",ln=True)

    return pdf.output(dest="S").encode("latin-1")

# ============================
# STREAMLIT UI
# ============================

st.title("Structural AI Crack Detection PRO v300")

file=st.file_uploader("Upload Image")

if file:

    img=cv2.imdecode(np.frombuffer(file.read(),np.uint8),1)

    width,length,area,severity,marked=detect(img)

    st.image(marked)

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Width",f"{width:.2f} mm")
    c2.metric("Length",f"{length:.2f} mm")
    c3.metric("Area",f"{area:.2f} mm²")
    c4.metric("Severity",severity)

    pdf=generate_pdf(width,length,area,severity)

    st.download_button("Download PDF",pdf,"report.pdf")

# ============================
# HISTORY
# ============================

if st.button("Show History"):

    conn=sqlite3.connect(DB)

    df=pd.read_sql("SELECT * FROM logs",conn)

    st.dataframe(df)

    conn.close()

# ============================
# CAMERA SUPPORT
# ============================

cam=st.camera_input("Live Camera")

if cam:

    img=cv2.imdecode(np.frombuffer(cam.read(),np.uint8),1)

    width,length,area,severity,marked=detect(img)

    st.image(marked)

    st.write(width,length,area,severity)



