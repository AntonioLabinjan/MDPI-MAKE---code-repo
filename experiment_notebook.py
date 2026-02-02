!pip install transformers
!pip install faiss-cpu  # Use faiss-gpu if you have a GPU in your environment
!pip install faiss-gpu
!pip install scikit-learn
!pip install matplotlib
!pip install seaborn
!pip install pillow

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

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')



# Set the dataset path from Google Drive
base_dir = "/content/drive/MyDrive/real_final"  # Replace with your dataset folder name

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP model loaded successfully!")

# Data placeholders
known_face_encodings = []
known_face_names = []

# Save the known face encodings and FAISS index (for faster execution time while experimenting with various parameters)
def save_known_faces_and_index():
    with open('known_face_encodings (5).pkl', 'wb') as f:
        pickle.dump(known_face_encodings, f)
    with open('known_face_names (5).pkl', 'wb') as f:
        pickle.dump(known_face_names, f)
    faiss.write_index(faiss_index, 'faiss_index.index')
    print("Known faces and FAISS index saved!")

# Load the known face encodings and FAISS index
def load_known_faces_and_index():
    global known_face_encodings, known_face_names, faiss_index
    if os.path.exists('known_face_encodings (5).pkl') and os.path.exists('faiss_index.index'):
        with open('known_face_encodings (5).pkl', 'rb') as f:
            known_face_encodings = pickle.load(f)
        with open('known_face_names (5).pkl', 'rb') as f:
            known_face_names = pickle.load(f)
        faiss_index = faiss.read_index('faiss_index.index')
        print("Loaded known faces and FAISS index from disk!")
    else:
        print("No saved data found. Proceeding with processing faces...")
        load_known_faces() # This line is added to populate the lists if no saved data is found
        faiss_index = build_index(known_face_encodings)
        save_known_faces_and_index()

# Add known face
def add_known_face(image_path, name):
    print(f"Processing image for known face: {image_path}")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    embedding = outputs.cpu().numpy().flatten()
    known_face_encodings.append(embedding / np.linalg.norm(embedding))
    known_face_names.append(name)
    print(f"Added known face: {name}")

# Load known faces from 'train' folders
def load_known_faces(base_dir=base_dir):
    print("Loading known faces...")
    for class_name in os.listdir(base_dir):
        train_dir = os.path.join(base_dir, class_name, "train")
        if os.path.isdir(train_dir):
            print(f"Processing class: {class_name}")
            for image_file in os.listdir(train_dir):
                image_path = os.path.join(train_dir, image_file)
                add_known_face(image_path, class_name)
    print("All known faces loaded successfully!")

# Build FAISS index
def build_index(encodings):
    print("Building FAISS index...")
    dimension = encodings[0].shape[0]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(encodings))
    print("FAISS index built successfully!")
    return faiss_index

from sklearn.metrics import accuracy_score
from itertools import product


def grid_search(val_dir, faiss_index, known_face_names, k1_values, k2_values, threshold_values, results_file="grid_search_results.txt"):
    best_params = {}
    best_accuracy = 0
    results = []  # Store detailed results

    # Write header to the results file
    with open(results_file, "w") as f:
        f.write("k1, k2, threshold, accuracy, recall, f1, tp, fp, tn, fn\n")

    print("Starting grid search...")
    for k1, k2, threshold in product(k1_values, k2_values, threshold_values):
        # Skip invalid combinations (ignore); it was used because Colab sometimes interrupted code execution in the middle of runs
        #if (k1 == 1 and k2 == 3) or (k1 == 1 and k2 == 6 and threshold < 0.8) or (k1 == 4 and k2 == 9 and threshold < 0.95) or (k1 == 6 and k2 == 7 and threshold < 0.65):
        #    continue
        print(f"\nTesting parameters: k1={k1}, k2={k2}, threshold={threshold}")

        # Classification function with current parameters
        def classify_with_params(face_embedding):
            return classify_face(face_embedding, faiss_index, known_face_names, k1=k1, k2=k2, threshold=threshold)

        # Evaluate the parameters
        true_labels, pred_labels = evaluate(val_dir, faiss_index, classify_with_params)

        if not true_labels or not pred_labels:  # Debug: Check if labels are empty
            print(f"[WARNING] No labels found for k1={k1}, k2={k2}, threshold={threshold}")
            continue

        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels, average="weighted", zero_division=0)
        f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

        cm = confusion_matrix(true_labels, pred_labels, labels=list(set(true_labels)))
        tp = np.diag(cm).sum()
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tn = cm.sum() - (fp + fn + tp)

        print(f"Metrics for k1={k1}, k2={k2}, threshold={threshold}:")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Recall: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - TP: {tp.sum()}, FP: {fp.sum()}, TN: {tn.sum()}, FN: {fn.sum()}")

        # Append results
        results.append({
            "k1": k1,
            "k2": k2,
            "threshold": threshold,
            "accuracy": accuracy,
            "recall": recall,
            "f1": f1,
            "tp": tp.sum(),
            "fp": fp.sum(),
            "tn": tn.sum(),
            "fn": fn.sum()
        })

        # Write results to file
        with open(results_file, "a") as f:
            row = f"{k1}, {k2}, {threshold}, {accuracy:.4f}, {recall:.4f}, {f1:.4f}, {tp.sum()}, {fp.sum()}, {tn.sum()}, {fn.sum()}\n"
            print(f"Writing to file: {row.strip()}")
            f.write(row)

        # Update best parameters if accuracy is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {"k1": k1, "k2": k2, "threshold": threshold}

    print("\nGrid search completed.")
    print(f"Best parameters: {best_params}")
    print(f"Best accuracy: {best_accuracy:.4f}")

    return best_params, best_accuracy, results


# Classify with FAISS


def classify_face(face_embedding, faiss_index, known_face_names, k1=3, k2=5, threshold=0.60): # this is default combination of k1, k2 and threshold
    D, I = faiss_index.search(face_embedding[np.newaxis, :], k2)
    votes = {}
    for idx, dist in zip(I[0][:k1], D[0][:k1]):
        if idx != -1 and dist <= threshold:
            label = known_face_names[idx]
            votes[label] = votes.get(label, 0) + 1
    if not votes:
        for idx, dist in zip(I[0][k1:], D[0][k1:]):
            if idx != -1 and dist <= threshold:
                label = known_face_names[idx]
                votes[label] = votes.get(label, 0) + 1
    majority_class = max(votes, key=votes.get) if votes else "Unknown"
    return majority_class, votes

# Evaluate on validation set

def evaluate(val_dir, faiss_index, classify_function):
    print("Evaluating on validation set...")
    true_labels = []
    pred_labels = []

    total_images = 0
    processed_images = 0

    for class_name in os.listdir(val_dir):
        val_class_dir = os.path.join(val_dir, class_name, "val")
        if os.path.isdir(val_class_dir):
            print(f"Processing class: {class_name}")
            for image_file in os.listdir(val_class_dir):
                total_images += 1
                image_path = os.path.join(val_class_dir, image_file)
                print(f"  - Loading image: {image_file} from class {class_name}")

                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    print(f"    [ERROR] Failed to load image {image_file}: {e}")
                    continue

                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.get_image_features(**inputs)
                face_embedding = outputs.cpu().numpy().flatten()

                face_embedding /= np.linalg.norm(face_embedding)
                print(f"    - Generated face embedding for {image_file}")

                pred_label, _ = classify_function(face_embedding)
                print(f"    - Predicted label: {pred_label}")

                true_labels.append(class_name)
                pred_labels.append(pred_label)

                processed_images += 1

    print(f"\nEvaluation completed: {processed_images}/{total_images} images processed.")
    return true_labels, pred_labels

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def calculate_metrics(true_labels, pred_labels):
    print("Calculating global metrics...\n")

    unique_labels = sorted(set(true_labels + pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)

    f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    recall = recall_score(true_labels, pred_labels, average="weighted", zero_division=0)
    accuracy = accuracy_score(true_labels, pred_labels)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (Weighted): {f1:.4f}")
    print(f"Recall (Weighted): {recall:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(true_labels, pred_labels, zero_division=0, digits=4))

    # === TP/FP/FN/TN per class ===
    print("\nPer-class TP, FP, FN, TN:")
    total = cm.sum()
    for i, label in enumerate(unique_labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        print(f"\nClass '{label}':")
        print(f"  TP = {tp}")
        print(f"  FP = {fp}")
        print(f"  FN = {fn}")
        print(f"  TN = {tn}")

    # === Visualization ===
    plt.figure(figsize=(10, 8), facecolor='white')
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=unique_labels, yticklabels=unique_labels,
                square=True, cbar=False)

    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix_visual.pdf", format="pdf", facecolor='white')
    plt.show()




# Main Workflow
if __name__ == "__main__":
    # Load known faces and build FAISS index if not saved
    load_known_faces_and_index()
    if not known_face_encodings:  # If there are no known faces loaded from disk
        load_known_faces()
        faiss_index = build_index(known_face_encodings)
        save_known_faces_and_index()  # Save for future use

    # Default parameters for initial evaluation
    default_k1 = 3
    default_k2 = 5
    default_threshold = 0.60

    # Evaluate on validation set with default parameters
    val_dir = os.path.join(base_dir)  # Ensure this points to the base directory
    true_labels, pred_labels = evaluate(
        val_dir,
        faiss_index,
        lambda face_embedding: classify_face(
            face_embedding, faiss_index, known_face_names, default_k1, default_k2, default_threshold
        )
    )

    # Define parameter ranges
    # Default values (placeholders)
    k1_values = [1]
    k2_values = [3]
    threshold_values = [0.82]

    # Run grid search
    best_params, best_accuracy, results = grid_search(
        val_dir=val_dir,
        faiss_index=faiss_index,
        known_face_names=known_face_names,
        k1_values=k1_values,
        k2_values=k2_values,
        threshold_values=threshold_values
    )


    # Output results
    print(f"Best parameters found: {best_params}")
    print(f"Best accuracy achieved: {best_accuracy}")

    # Calculate metrics
    calculate_metrics(true_labels, pred_labels)

