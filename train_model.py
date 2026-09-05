"""
train_model.py
---------------
Step 2 of the pipeline. Reads every image under dataset/<user_id>_<name>/,
trains an LBPH (Local Binary Patterns Histogram) face recognizer, and
saves the trained model to trainer.yml. Run this again any time you add
a new user or more samples for an existing one.

Usage:
    python train_model.py
"""

import os

import paths
import cv2
import numpy as np

DATASET_DIR = paths.dataset_dir()
MODEL_PATH = paths.model_path()


def load_training_data():
    faces = []
    labels = []

    if not os.path.isdir(DATASET_DIR):
        return faces, labels

    for folder_name in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # folder_name looks like "3_Jane_Doe" -> label/user_id is 3
        try:
            user_id = int(folder_name.split("_")[0])
        except ValueError:
            print(f"Skipping folder with unexpected name: {folder_name}")
            continue

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(user_id)

    return faces, labels


def train():
    faces, labels = load_training_data()

    if len(faces) == 0:
        print("No training images found. Run register_user.py first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)

    unique_users = len(set(labels))
    print(f"Trained on {len(faces)} images across {unique_users} user(s).")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
