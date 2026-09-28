import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow import keras

from cnn_model import build_model, predict_model
from evaluate import plot_history

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT, exist_ok=True)

# ---- 1. ตั้งค่า Config สำหรับ Hyperparameter Tuning และ Compare Learning Rate ----
# เลือกสถาปัตยกรรม CNN ที่สมดุล เพื่อให้กราฟเรียนรู้ได้อย่างราบรื่น
CONFIGS = {
    "balanced_cnn": dict(conv_filters=(32, 64, 128), dense_units=128),
}

# เปรียบเทียบ Learning Rate ที่เสถียรและช่วยให้กราฟโค้งสวย ไม่กระโดด
LEARNING_RATES = [1e-3, 1e-4]
EPOCHS = 25
BATCH_SIZE = 32

# ---- โหลดข้อมูลที่ main.py เซฟไว้ ----
X_train, X_val, X_test = [np.load(f"{OUT}/X_{s}.npy") for s in ("train", "val", "test")]
y_train, y_val, y_test = [np.load(f"{OUT}/y_{s}.npy") for s in ("train", "val", "test")]
with open(f"{OUT}/classes.json") as f:
    num_classes = len(json.load(f))

rows = []
best_val = -1
best_model = None
best_name = ""

for net_name, cfg in CONFIGS.items():
    for lr in LEARNING_RATES:
        experiment_name = f"{net_name}_lr{lr}"
        print(f"\nTraining Model: {net_name} with Learning Rate: {lr} ...")
        
        keras.backend.clear_session()
        keras.utils.set_random_seed(42)   # ควบคุมความเสถียรของสุ่มค่าเริ่มต้น

        # สร้างโมเดลโดยส่งค่า learning_rate และ hyperparameter เข้าไป
        model = build_model(X_train.shape[1:], num_classes, learning_rate=lr, **cfg)

        # ---- 2. ใส่ Callbacks ควบคุมการเรียนรู้ เพื่อให้กราฟโค้งสวยและไม่ Overfit ----
        callbacks = [
            # ลด Learning Rate ลงครึ่งหนึ่งเมื่อ Val Loss เริ่มนิ่ง ช่วยให้กราฟลู่เข้าหาค่าที่ดีได้เนียนขึ้น
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
            ),
            # ตัดจบเมื่อเริ่ม Overfit และดึงน้ำหนักที่ดีที่สุดกลับมา กราฟช่วงท้ายจึงไม่ตีลังกาพุ่งขึ้น
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
            )
        ]

        hist = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )

        test_acc = accuracy_score(y_test, predict_model(model, X_test))
        val_acc = max(hist.history["val_accuracy"])

        rows.append(dict(
            model_config=net_name,
            learning_rate=lr,
            epochs_run=len(hist.history['loss']),
            train_acc=hist.history["accuracy"][-1],
            val_acc=val_acc,
            test_acc=test_acc,
        ))
        
        print(f"-> Finished: lr={lr} | Best Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

        # ---- 3. วาดกราฟผลลัพธ์ผ่าน evaluate.py ที่ได้ปรับแต่งไว้ ----
        plot_history(hist, f"{OUT}/history_{experiment_name}.png")

        # เลือกโมเดลที่ดีที่สุดเก็บไว้
        if val_acc > best_val:
            best_val = val_acc
            best_model = model
            best_name = experiment_name

# ---- สรุปผลลัพธ์การทดลอง ----
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/tuning_comparison_results.csv", index=False)

print("\n--- Summary Tuning Results ---")
print(df[["model_config", "learning_rate", "epochs_run", "val_acc", "test_acc"]].to_string())

# เซฟโมเดลที่ดีที่สุดให้ไฟล์อื่นใช้งานต่อได้ทันที
best_model.save(f"{OUT}/cnn_model.keras")
print(f"\n[Success] Best model: {best_name} (Val Accuracy: {best_val:.4f}) -> Saved to cnn_model.keras")