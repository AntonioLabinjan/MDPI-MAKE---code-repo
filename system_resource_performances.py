# Instalacija potrebnih paketa
!pip install transformers faiss-cpu scikit-learn matplotlib seaborn pillow tqdm psutil

import os
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pickle
from itertools import product
import psutil
import time
from tqdm.auto import tqdm  # Progress bar biblioteka

# --- 1. POSTAVKE I MOUNT ---
print("\n[STEP 1] Mounting Google Drive...")
from google.colab import drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

base_dir = "/content/drive/MyDrive/real_final"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")

# --- 2. FUNKCIJE ZA RESURSE ---

def get_resource_usage():
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)
    cpu_load = psutil.cpu_percent(interval=None)
    gpu_mem = 0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)
    return ram_usage, cpu_load, gpu_mem

# --- 3. MODEL I FAISS ---

print("\n[STEP 2] Loading CLIP model from OpenAI...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Model loaded successfully!")

known_face_encodings = []
known_face_names = []

def build_index(encodings):
    print("\n[PHASE] Building FAISS Index...")
    dimension = encodings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(encodings).astype('float32'))
    print(f"Index built with {len(encodings)} vectors.")
    return index

def save_known_faces_and_index(index):
    print("\n[STORAGE] Saving data to disk...")
    with open('known_face_encodings.pkl', 'wb') as f:
        pickle.dump(known_face_encodings, f)
    with open('known_face_names.pkl', 'wb') as f:
        pickle.dump(known_face_names, f)
    faiss.write_index(index, 'faiss_index.index')
    print("Files 'known_face_encodings.pkl', 'known_face_names.pkl' and 'faiss_index.index' saved!")

# --- 4. CORE LOGIKA ---

def get_embedding(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        embedding_tensor = getattr(outputs, "pooler_output", outputs if isinstance(outputs, torch.Tensor) else outputs[0])
        emb = embedding_tensor.cpu().numpy().flatten()
        return emb / (np.linalg.norm(emb) + 1e-10)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def load_known_faces():
    print("\n[PHASE] Processing TRAIN dataset...")
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for class_name in tqdm(classes, desc="Overall Class Progress"):
        train_dir = os.path.join(base_dir, class_name, "train")
        if os.path.isdir(train_dir):
            img_files = [f for f in os.listdir(train_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for image_file in img_files:
                emb = get_embedding(os.path.join(train_dir, image_file))
                if emb is not None:
                    known_face_encodings.append(emb)
                    known_face_names.append(class_name)

def classify_face(face_embedding, faiss_index, known_names, k1=3, k2=5, threshold=0.60):
    D, I = faiss_index.search(face_embedding[np.newaxis, :].astype('float32'), k2)
    votes = {}
    for idx, dist in zip(I[0][:k1], D[0][:k1]):
        if idx != -1 and dist <= threshold:
            label = known_names[idx]
            votes[label] = votes.get(label, 0) + 1
    if not votes:
        for idx, dist in zip(I[0][k1:], D[0][k1:]):
            if idx != -1 and dist <= threshold:
                label = known_names[idx]
                votes[label] = votes.get(label, 0) + 1
    return max(votes, key=votes.get) if votes else "Unknown", votes

# --- 5. EVALUACIJA S PROFILIRANJEM ---

def evaluate_with_metrics(val_dir, faiss_index, k1, k2, threshold):
    print(f"\n[PHASE] Starting Evaluation (k1={k1}, k2={k2}, thresh={threshold})...")
    true_labels, pred_labels = [], []
    latencies, ram_usage, cpu_usage = [], [], []
    
    psutil.cpu_percent(interval=None) # reset

    classes = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    
    # Brojimo ukupno slika za progress bar
    total_images = 0
    for c in classes:
        v_path = os.path.join(val_dir, c, "val")
        if os.path.isdir(v_path):
            total_images += len([f for f in os.listdir(v_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    pbar = tqdm(total=total_images, desc="Evaluating Images")

    for class_name in classes:
        val_class_dir = os.path.join(val_dir, class_name, "val")
        if os.path.isdir(val_class_dir):
            img_files = [f for f in os.listdir(val_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for image_file in img_files:
                img_path = os.path.join(val_class_dir, image_file)
                
                start_time = time.perf_counter()
                emb = get_embedding(img_path)
                
                if emb is not None:
                    pred_label, _ = classify_face(emb, faiss_index, known_face_names, k1, k2, threshold)
                    end_time = time.perf_counter()
                    r, c, g = get_resource_usage()
                    
                    latencies.append((end_time - start_time) * 1000)
                    ram_usage.append(r)
                    cpu_usage.append(c)
                    
                    true_labels.append(class_name)
                    pred_labels.append(pred_label)
                
                pbar.update(1)
    
    pbar.close()

    stats = {
        "avg_latency": np.mean(latencies),
        "avg_ram": np.mean(ram_usage),
        "avg_cpu": np.mean(cpu_usage),
        "gpu_final": g,
        "accuracy": accuracy_score(true_labels, pred_labels)
    }
    return true_labels, pred_labels, stats

# --- 6. MAIN IZVRŠAVANJE ---

if __name__ == "__main__":
    # Inicijalizacija baze
    if os.path.exists('faiss_index.index'):
        print("\n[INFO] Found existing FAISS index on disk. Loading...")
        faiss_index = faiss.read_index('faiss_index.index')
        with open('known_face_names.pkl', 'rb') as f:
            known_face_names = pickle.load(f)
        print(f"Loaded index with {faiss_index.ntotal} faces.")
    else:
        print("\n[INFO] No existing index found. Starting fresh.")
        load_known_faces()
        faiss_index = build_index(known_face_encodings)
        save_known_faces_and_index(faiss_index)

    # Parametri
    k1, k2, threshold = 1, 3, 0.82
    
    # Eval
    t_labels, p_labels, profiling_stats = evaluate_with_metrics(base_dir, faiss_index, k1, k2, threshold)

    # FINALNI REPORT
    print("\n" + "█"*50)
    print("      FINAL RESOURCE & ACCURACY REPORT")
    print("█"*50)
    print(f"  Accuracy:           {profiling_stats['accuracy']*100:.2f}%")
    print(f"  Avg Inference Time: {profiling_stats['avg_latency']:.2f} ms")
    print(f"  Avg RAM Usage:      {profiling_stats['avg_ram']:.2f} MB")
    print(f"  Avg CPU Load:       {profiling_stats['avg_cpu']:.2f} %")
    print(f"  Peak GPU Memory:    {profiling_stats['gpu_final']:.2f} MB")
    print("█"*50)

    print("\n[DETAILED REPORT]")
    print(classification_report(t_labels, p_labels))
