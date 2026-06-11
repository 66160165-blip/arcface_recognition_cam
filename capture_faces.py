import os
import cv2
import numpy as np

# ==========================================
# CONFIG
# ==========================================
DATASET_RAW = "dataset_raw"
os.makedirs(DATASET_RAW, exist_ok=True)

# กลับภาพซ้าย-ขวา (เหมือนกระจก)
# True  = ขยับซ้าย ภาพไปซ้าย (ใช้งานง่าย)
# False = ไม่กลับภาพ
MIRROR_VIEW = True

# เปิด Sharpen เพื่อเพิ่มความคมชัด
ENABLE_SHARPEN = True

# ความละเอียดกล้อง
CAM_WIDTH = 2560
CAM_HEIGHT = 1080
CAM_FPS = 60

# ==========================================
# INPUT PERSON NAME
# ==========================================
person_name = input("Enter person name (Student ID): ").strip()

if not person_name:
    print("❌ Person name cannot be empty.")
    exit()

# ==========================================
# CREATE PERSON FOLDER
# ==========================================
person_dir = os.path.join(DATASET_RAW, person_name)
os.makedirs(person_dir, exist_ok=True)

# ==========================================
# FIND NEXT IMAGE NUMBER
# ==========================================
existing_files = [
    f for f in os.listdir(person_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

numbers = []
for f in existing_files:
    name, _ = os.path.splitext(f)
    if name.isdigit():
        numbers.append(int(name))

next_number = max(numbers) + 1 if numbers else 1

# ==========================================
# INFO
# ==========================================
print("=" * 60)
print(f"📁 Saving images to: {person_dir}")
print(f"📸 Next image number: {next_number}")
print("⌨️ Controls:")
print("   SPACE = Capture image")
print("   ESC   = Exit")
print("=" * 60)

# ==========================================
# OPEN CAMERA
# ==========================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera.")
    exit()

# ตั้งค่าความละเอียด
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

# แสดงค่าที่กล้องตั้งได้จริง
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"📷 Camera Resolution: {actual_width} x {actual_height}")
print(f"🎞️ FPS: {actual_fps}")

# Kernel สำหรับ Sharpen
sharpen_kernel = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
])

saved_count = 0

# ==========================================
# MAIN LOOP
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read frame.")
        break

    # --------------------------------------
    # Mirror View
    # --------------------------------------
    if MIRROR_VIEW:
        frame = cv2.flip(frame, 1)

    # --------------------------------------
    # Sharpen
    # --------------------------------------
    if ENABLE_SHARPEN:
        frame = cv2.filter2D(frame, -1, sharpen_kernel)

    # --------------------------------------
    # Display Copy
    # --------------------------------------
    display = frame.copy()

    cv2.putText(
        display,
        f"Person: {person_name}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Saved: {saved_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Resolution: {actual_width}x{actual_height}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        "SPACE = Capture | ESC = Exit",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Capture Faces", display)

    key = cv2.waitKey(1) & 0xFF

    # ESC = Exit
    if key == 27:
        break

    # SPACE = Capture
    elif key == 32:
        filename = f"{next_number}.jpg"
        filepath = os.path.join(person_dir, filename)

        # บันทึกภาพจริง (รวม mirror + sharpen)
        cv2.imwrite(filepath, frame)

        print(f"✅ Saved: {filepath}")

        next_number += 1
        saved_count += 1

# ==========================================
# CLEANUP
# ==========================================
cap.release()
cv2.destroyAllWindows()

print("=" * 60)
print(f"🎉 Done! Total images captured: {saved_count}")
print(f"📁 Images stored in: {person_dir}")
print("=" * 60)