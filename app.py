import io
import os
import json
import uuid
import numpy as np
import joblib
import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fpdf import FPDF

# =====================================================================
# 1. FIXED SETTINGS
# =====================================================================
MAP_ORDER = ["CT_A", "EC_A", "EC_P", "ELV_A", "ELV_P", "SAG_A", "SAG_P"]
CLASS_NAMES = {0: "Normal", 1: "Keratoconus", 2: "Suspect"}
TEMP_DIR = "temp_records"
os.makedirs(TEMP_DIR, exist_ok=True)

# =====================================================================
# 2. LOAD ML BUNDLE
# =====================================================================
bundle  = joblib.load("model.pkl")
svm      = bundle["svm"]
scaler   = bundle["scaler"]
selector = bundle["selector"]

# =====================================================================
# 3. LOAD PYTORCH MODEL & PREPROCESSING
# =====================================================================
device = torch.device("cpu")
densenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
densenet.classifier = nn.Identity()
densenet = densenet.to(device).eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def preprocess_kaggle_style(img_array: np.ndarray) -> Image.Image:
    """Replicates the exact preprocessing from your Kaggle notebook."""
    # 1. Convert BGR to RGB
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # 2. Crop right side (remove text)
    crop_w = int(w * 0.82)
    img = img[:, :crop_w, :]
    
    # 3. Crop top and bottom
    crop_h_top = int(h * 0.08)
    crop_h_bottom = int(h * 0.90)
    img = img[crop_h_top:crop_h_bottom, :, :]
    
    # 4. Find largest contour (the eye)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        largest_idx = np.argmax(areas)
        x, y, w_box, h_box = cv2.boundingRect(contours[largest_idx])
        pad = 5
        y1, y2 = max(0, y - pad), min(h, y + h_box + pad)
        x1, x2 = max(0, x - pad), min(w, x + w_box + pad)
        img = img[y1:y2, x1:x2]
        
    # 5. Resize to 224x224
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    
    # 6. Convert to Grayscale and back to RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(img)

def extract_feature(raw_bytes: bytes):
    # Read bytes as OpenCV image
    nparr = np.frombuffer(raw_bytes, np.uint8)
    img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Apply Kaggle preprocessing
    img = preprocess_kaggle_style(img_array)
    
    # PyTorch Transform
    x = transform(img).unsqueeze(0).to(device)
    
    # Saliency Map generation
    x.requires_grad_()
    with torch.enable_grad():
        feat = densenet(x)
        feat.sum().backward()
        saliency, _ = torch.max(x.grad.data.abs(), dim=1)
        saliency = saliency.squeeze().cpu().numpy()
    
    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    # --- GRAD-CAM STYLE REFINEMENT ---
    # 1. Apply Heavy Gaussian Blur to make smooth blobs (removes pixel noise)
    saliency = cv2.GaussianBlur(saliency, (21, 21), 0)
    
    # Re-normalize after blur
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    # 2. Create a mask: Only keep the top 30% most important areas (threshold = 0.7)
    mask = saliency > 0.7
    
    # 3. Convert to JET heatmap (Blue/Green/Red)
    saliency_uint8 = (saliency * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 4. Blend ONLY the hotspots over the original image
    orig_np = np.array(img)
    blended = orig_np.copy()
    
    # Where the mask is True, blend 40% original + 60% heatmap
    blended[mask] = cv2.addWeighted(orig_np, 0.4, heatmap, 0.6, 0)[mask]
    
    saliency_img = Image.fromarray(blended)
    
    return feat.detach().cpu().numpy()[0], saliency_img


# =====================================================================
# 4. THE API
# =====================================================================
app = FastAPI(title="Keratoconus Multi-Map Classifier")

@app.post("/predict")
async def predict(
    name: str = Form(...),
    age: str = Form(...),
    vision: str = Form(...),
    SAG_A: UploadFile = File(...), SAG_P: UploadFile = File(...),
    ELV_A: UploadFile = File(...), ELV_P: UploadFile = File(...),
    CT_A:  UploadFile = File(...),
    EC_A:  UploadFile = File(...), EC_P:  UploadFile = File(...),
):
    uploads = {"SAG_A": SAG_A, "SAG_P": SAG_P, "ELV_A": ELV_A,
               "ELV_P": ELV_P, "CT_A": CT_A, "EC_A": EC_A, "EC_P": EC_P}

    # Create a folder for this patient to save images and PDF later
    record_id = str(uuid.uuid4())[:8]
    record_dir = os.path.join(TEMP_DIR, record_id)
    os.makedirs(record_dir, exist_ok=True)

    feats = []
    saliency_maps = {}
    
    for map_name in MAP_ORDER:
        raw = await uploads[map_name].read()
        
        # Save original image
        with open(os.path.join(record_dir, f"{map_name}.png"), "wb") as f:
            f.write(raw)
            
        feat, saliency = extract_feature(raw)
        feats.append(feat)
        
        # Save saliency map
        sal_path = os.path.join(record_dir, f"{map_name}_saliency.png")
        saliency.save(sal_path)
        saliency_maps[map_name] = sal_path

    x = np.concatenate(feats).reshape(1, -1)
    x_scaled   = scaler.transform(x)
    x_selected = selector.transform(x_scaled)

    pred_idx = int(svm.predict(x_selected)[0])
    result = {"predicted_class": CLASS_NAMES[pred_idx], "record_id": record_id}

    try:
        probs = svm.predict_proba(x_selected)[0]
        result["probabilities"] = {
            CLASS_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(probs)
        }
    except AttributeError:
        pass

    # Save patient info to JSON for the PDF generator
    with open(os.path.join(record_dir, "info.json"), "w") as f:
        json.dump({
            "name": name, "age": age, "vision": vision,
            "prediction": CLASS_NAMES[pred_idx],
            "probabilities": result.get("probabilities", {})
        }, f)

    return result

@app.get("/download_report/{record_id}")
def download_report(record_id: str):
    record_dir = os.path.join(TEMP_DIR, record_id)
    if not os.path.exists(record_dir):
        return {"error": "Record not found"}

    with open(os.path.join(record_dir, "info.json"), "r") as f:
        info = json.load(f)

    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(200, 10, text="Keratoconus Screening Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(10)

    pdf.set_font("Helvetica", size=12)
    pdf.cell(200, 10, text=f"Patient Name: {info['name']}")
    pdf.ln(8)
    pdf.cell(200, 10, text=f"Age: {info['age']}")
    pdf.ln(8)
    pdf.cell(200, 10, text=f"Vision (Acuity): {info['vision']}")
    pdf.ln(8)
    pdf.cell(200, 10, text=f"AI Prediction: {info['prediction']}")
    pdf.ln(10)

    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(200, 10, text="Class Probabilities (%):")
    pdf.ln(8)
    pdf.set_font("Helvetica", size=12)
    for cls, prob in info['probabilities'].items():
        pdf.cell(200, 10, text=f"  - {cls}: {prob}%")
        pdf.ln(8)

    pdf.ln(10)
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(200, 10, text="Pentacam Maps & AI Heatmaps (Focus Areas):")
    pdf.ln(10)

    # Structured Grid Layout
    for map_name in MAP_ORDER:
        img_path = os.path.join(record_dir, f"{map_name}.png")
        sal_path = os.path.join(record_dir, f"{map_name}_saliency.png")
        
        # If we are near the bottom of the page, add a new page
        if pdf.get_y() > 230:
            pdf.add_page()
            
        current_y = pdf.get_y()
        
        # Column 1: Original Image Label
        pdf.set_font("Helvetica", 'B', 10)
        pdf.set_xy(10, current_y)
        pdf.cell(90, 8, text=f"{map_name} (Original)", border=0, align="L")
        
        # Column 2: AI Focus Label
        pdf.set_xy(110, current_y)
        pdf.cell(90, 8, text=f"{map_name} (AI Focus)", border=0, align="L")
        
        # Place Images strictly using x, y, w, h
        img_y = current_y + 8
        pdf.image(img_path, x=10, y=img_y, w=80, h=60)
        pdf.image(sal_path, x=110, y=img_y, w=80, h=60)
        
        # Draw borders around images for a clean medical report look
        pdf.rect(10, img_y, 80, 60)
        pdf.rect(110, img_y, 80, 60)
        
        # Move Y down for the next row (Image height + spacing)
        pdf.set_xy(10, img_y + 65)

    pdf_path = os.path.join(record_dir, "report.pdf")
    pdf.output(pdf_path)
    
    return FileResponse(pdf_path, media_type='application/pdf', filename=f"Keratoconus_Report_{info['name']}.pdf")

# =====================================================================
# 5. SERVE THE FRONTEND
# =====================================================================
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")