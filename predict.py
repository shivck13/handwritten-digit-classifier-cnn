import cv2
from tensorflow.keras.models import load_model

# ---------------- Configurations ----------------
MODEL_PATH = "model.h5"
IMG_SIZE = 28
MIN_CONTOUR_AREA = 300
MAX_CONTOUR_AREA = 8000

# window titles
WINDOW_WEBCAM = "Digit Prediction from Live Camera"
WINDOW_IMAGE  = "Digit Prediction from Image File"

# Load CNN model
model = load_model(MODEL_PATH, compile=False)

# ---------------- Helper Functions ----------------
def preprocess_image(image):
    """Convert to grayscale, denoise, threshold, and morphologically clean."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.dilate(cleaned, kernel, iterations=2)
    return cleaned

# Finding contours and extracting bounding boxes
def extract_digits(thresh):
    """Extract bounding boxes of potential digit contours."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [(x, y, w, h) for cnt in contours
             if MIN_CONTOUR_AREA < cv2.contourArea(cnt) < MAX_CONTOUR_AREA
             for x, y, w, h in [cv2.boundingRect(cnt)]]
    return sorted(boxes, key=lambda b: b[0])  # Left-to-right

# Resize and normalize ROI for model prediction
def prepare_image(roi):
    """Prepare ROI for CNN prediction."""
    resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

def predict_digit(roi):
    """Predict digit and confidence from ROI."""
    img = prepare_image(roi)
    preds = model.predict(img, verbose=0)
    digit = int(preds.argmax())
    confidence = float(preds[0][digit])
    return digit, confidence

def annotate_frame(frame, boxes, thresh):
    """Predict digits in bounding boxes and annotate frame."""
    vis = frame.copy()
    for (x, y, w, h) in boxes:
        roi = thresh[y:y+h, x:x+w]
        digit, conf = predict_digit(roi)
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{digit} ({conf*100:.0f}%)"
        cv2.putText(vis, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return vis

# ---------------- Main Functions ----------------
def predict_from_webcam():
    """Capture video from webcam and predict digits in real-time."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        thresh = preprocess_image(frame)
        boxes = extract_digits(thresh)
        vis_live = annotate_frame(frame, boxes, thresh)
        cv2.imshow(WINDOW_WEBCAM, vis_live)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(WINDOW_WEBCAM, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

def predict_from_image_file(image_path: str):
    """Predict digits from a static image file."""
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image file: {image_path}")

    # Resize if too large
    max_wh = 1024
    height, width = frame.shape[:2]
    if max(height, width) > max_wh:
        scale = max_wh / max(height, width)
        frame = cv2.resize(frame, (int(width*scale), int(height*scale)))

    thresh = preprocess_image(frame)
    boxes = extract_digits(thresh)
    vis = annotate_frame(frame, boxes, thresh)
    cv2.imshow(WINDOW_IMAGE, vis)
    cv2.waitKey(0) # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()
