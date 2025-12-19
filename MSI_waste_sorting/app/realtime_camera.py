import cv2
import joblib
import numpy as np
from skimage.feature import hog, local_binary_pattern

# ================= CONFIG =================
# Must match the settings used during training in 03_features.ipynb
IMG_SIZE = (128, 128)
HOG_ORIENTATIONS = 18
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (3, 3)
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CONF_THRESHOLD = 0.6
MAX_FRAMES = 1000
# ==========================================

# Load models
try:
    svm = joblib.load("../models/svm_best.pkl")
    scaler = joblib.load("../models/scaler_knn.pkl")
    print("Models loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    exit()


def extract_hog(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm='L2-Hys'
    )
    return features


def extract_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist


def extract_color(img):
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def extract_features(frame):
    img = cv2.resize(frame, IMG_SIZE)
    h_feats = extract_hog(img)
    c_feats = extract_color(img)
    l_feats = extract_lbp(img)

    # Concatenate exactly as done in 03_features.ipynb (HOG + Color + LBP)
    combined = np.concatenate([h_feats, c_feats, l_feats])
    return combined.reshape(1, -1)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found")
    exit()

print("Press 'Q' to quit, 'S' to save frame")

frame_count = 0
feature_mismatch_warned = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_count += 1
        if frame_count > MAX_FRAMES:
            print("Reached max frames, exiting...")
            break

        # Extract combined features
        feats = extract_features(frame)

        # Check if features match scaler input (Expected: 6354)
        if feats.shape[1] != scaler.n_features_in_:
            label = "UNKNOWN"
            color = (0, 0, 255)
            if not feature_mismatch_warned:
                print(f"Feature size mismatch: got {feats.shape[1]}, expected {scaler.n_features_in_}")
                feature_mismatch_warned = True
        else:
            feats_scaled = scaler.transform(feats)
            probs = svm.predict_proba(feats_scaled)[0]
            max_prob = np.max(probs)
            pred = np.argmax(probs)

            if max_prob < CONF_THRESHOLD:
                label = "UNKNOWN"
                color = (0, 0, 255)
            else:
                label = CLASSES[pred]
                color = (0, 255, 0)

        # UI Overlay
        display_text = f"Prediction: {label} ({max_prob:.2f})" if label != "UNKNOWN" else "UNKNOWN"
        cv2.putText(
            frame,
            display_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.imshow("Waste Sorting Identification", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            filename = f"capture_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released.")