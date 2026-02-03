#### RUNNING THIS ONE

import os
import numpy as np
import torch
import faiss
import pickle
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torchvision.models as models
import torchvision.transforms as T
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from tqdm.auto import tqdm  # Progress bar biblioteka

# Mount Drive
from google.colab import drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# Putanja
base_dir = "/content/drive/MyDrive/real_final"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. LOAD MODELS (CLIP & RESNET) ---
print("Loading Models...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
resnet_model.eval()
resnet_extractor = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))
resnet_preprocess = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print("Models loaded successfully!")

# --- 2. EMBEDDING HELPER FUNCTIONS ---

def get_clip_embedding(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)
        tensor = getattr(outputs, "pooler_output", outputs if isinstance(outputs, torch.Tensor) else outputs[0])
        embedding = tensor.detach().cpu().numpy().flatten()
        return embedding / (np.linalg.norm(embedding) + 1e-10)
    except Exception as e:
        return None

def get_resnet_embedding(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        img_t = resnet_preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = resnet_extractor(img_t)
        embedding = features.detach().cpu().numpy().flatten()
        return embedding / (np.linalg.norm(embedding) + 1e-10)
    except Exception as e:
        return None

# --- 3. DATA PROCESSING ---

def process_dataset(model_type="clip"):
    encodings, names = [], []
    print(f"\n[PHASE] Processing TRAIN dataset for {model_type.upper()}")
    
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # Progress bar za klase
    for class_name in tqdm(classes, desc="Overall Progress"):
        train_dir = os.path.join(base_dir, class_name, "train")
        if os.path.isdir(train_dir):
            img_files = os.listdir(train_dir)
            # Unutarnji progress bar za slike u klasi (opcionalno, ako ih ima puno)
            for img_file in img_files:
                path = os.path.join(train_dir, img_file)
                emb = get_clip_embedding(path) if model_type == "clip" else get_resnet_embedding(path)
                if emb is not None:
                    encodings.append(emb)
                    names.append(class_name)
    return np.array(encodings).astype('float32'), names

# --- 4. EVALUATION LOGIC ---

def evaluate_model(val_dir, index, known_names, model_type="clip"):
    true_labels, pred_labels = [], []
    print(f"\n[PHASE] Evaluating {model_type.upper()} on Validation Set")
    
    classes = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    
    for class_name in tqdm(classes, desc="Evaluating Classes"):
        val_class_dir = os.path.join(val_dir, class_name, "val")
        if os.path.isdir(val_class_dir):
            for img_file in os.listdir(val_class_dir):
                path = os.path.join(val_class_dir, img_file)
                emb = get_clip_embedding(path) if model_type == "clip" else get_resnet_embedding(path)
                
                if emb is not None:
                    # Search
                    D, I = index.search(emb.reshape(1, -1), k=1)
                    pred_label = known_names[I[0][0]] if I[0][0] != -1 else "Unknown"
                    
                    true_labels.append(class_name)
                    pred_labels.append(pred_label)
    return true_labels, pred_labels

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # CLIP Phase
    clip_encodings, clip_names = process_dataset("clip")
    clip_index = faiss.IndexFlatL2(512)
    clip_index.add(clip_encodings)
    clip_true, clip_pred = evaluate_model(base_dir, clip_index, clip_names, "clip")
    
    # ResNet Phase
    resnet_encodings, resnet_names = process_dataset("resnet")
    resnet_index = faiss.IndexFlatL2(2048)
    resnet_index.add(resnet_encodings)
    res_true, res_pred = evaluate_model(base_dir, resnet_index, resnet_names, "resnet")

    # Final Results Table
    print("\n" + "="*50)
    print(f"{'MODEL':<20} | {'ACCURACY':<12} | {'F1-SCORE':<10}")
    print("-" * 50)
    print(f"{'CLIP (Proposed)':<20} | {accuracy_score(clip_true, clip_pred):.4f} | {f1_score(clip_true, clip_pred, average='weighted'):.4f}")
    print(f"{'ResNet50 (Baseline)':<20} | {accuracy_score(res_true, res_pred):.4f} | {f1_score(res_true, res_pred, average='weighted'):.4f}")
    print("="*50)
