import io
import os
import json
import uuid
import gc
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

def preprocess_and_crop(img_array: np.ndarray):
    """Crops the eye, resizes, and returns both color and grayscale 3-channel arrays."""
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    crop_w = int(w * 0.82)
    img = img[:, :crop_w, :]
    
    crop_h_top = int(h * 0.08)
    crop_h_bottom = int(h * 0.90)
    img = img[crop_h_top:crop_h_bottom, :, :]
    
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
        
    img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    img_gray_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    
    return img_resized, img_gray_rgb

def process_images_sequentially(raw_bytes_list: list):
    """Processes images one by one to save RAM. Only generates 1 heatmap."""
    color_imgs = []
    heatmaps = []
    feats_np = []
    
    for i, raw in enumerate(raw_bytes_list):
        nparr = np.frombuffer(raw, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        color_img, ml_img = preprocess_and_crop(img_array)
        color_imgs.append(color_img)
        
        pil_img = Image.fromarray(ml_img)
        x = transform(pil_img).unsqueeze(0).to(device)
        del pil_img, ml_img, img_array, nparr
        
        # 1. Fast forward pass (NO gradients) to get features
        with torch.no_grad():
            feat = densenet(x)
        feats_np.append(feat.cpu().numpy()[0])
        del feat

        # 2. Saliency map (WITH gradients) - ONLY for the first image (CT_A)
        if i == 0:
            x.requires_grad_()
            with torch.enable_grad():
                feat_grad = densenet(x)
                feat_grad.sum().backward()
                saliency, _ = torch.max(x.grad.data.abs(), dim=1)
                saliency = saliency.squeeze().cpu().numpy()
            
            # Process heatmap
            sal = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            sal = cv2.GaussianBlur(sal, (21, 21), 0)
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
            
            mask = sal > 0.5
            sal_uint8 = (sal * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(sal_uint8, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            blended = color_img.copy()
            blended[mask] = cv2.addWeighted(color_img, 0.3, heatmap, 0.7, 0)[mask]
            heatmaps.append(Image.fromarray(blended))
            
            # Aggressive memory cleanup
            del x, feat_grad, saliency, sal, mask, sal_uint8, heatmap, blended
            gc.collect()
        else:
            # For the other 6 images, we don't generate a heatmap to save RAM
            heatmaps.append(None)
            del x
            gc.collect()
        
    return np.array(feats_np), color_imgs, heatmaps

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

    record_id = str(uuid.uuid4())[:8]
    record_dir = os.path.join(TEMP_DIR, record_id)
    os.makedirs(record_dir, exist_ok=True)

    raw_bytes_dict = {}
    for map_name in MAP_ORDER:
        raw_bytes_dict[map_name] = await uploads[map_name].read()
        
    raw_list = [raw_bytes_dict[name] for name in MAP_ORDER]
    feats_np, color_imgs, heatmaps = process_images_sequentially(raw_list)
    
    for i, map_name in enumerate(MAP_ORDER):
        Image.fromarray(color_imgs[i]).save(os.path.join(record_dir, f"{map_name}.png"))
        if heatmaps[i] is not None:
            heatmaps[i].save(os.path.join(record_dir, f"{map_name}_saliency.png"))

    x = np.concatenate([feats_np[i] for i in range(7)]).reshape(1, -1)
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

    for map_name in MAP_ORDER:
        img_path = os.path.join(record_dir, f"{map_name}.png")
        sal_path = os.path.join(record_dir, f"{map_name}_saliency.png")
        
        if pdf.get_y() > 230:
            pdf.add_page()
            
        current_y = pdf.get_y()
        
        pdf.set_font("Helvetica", 'B', 10)
        pdf.set_xy(10, current_y)
        pdf.cell(90, 8, text=f"{map_name} (Original)", border=0, align="L")
        
        pdf.set_xy(110, current_y)
        if os.path.exists(sal_path):
            pdf.cell(90, 8, text=f"{map_name} (AI Focus)", border=0, align="L")
        else:
            pdf.cell(90, 8, text=f"{map_name} (Feature Only)", border=0, align="L")
        
        img_y = current_y + 8
        pdf.image(img_path, x=10, y=img_y, w=80, h=60)
        pdf.rect(10, img_y, 80, 60)
        
        if os.path.exists(sal_path):
            pdf.image(sal_path, x=110, y=img_y, w=80, h=60)
            pdf.rect(110, img_y, 80, 60)
        else:
            pdf.rect(110, img_y, 80, 60)
        
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