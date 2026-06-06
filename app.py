import os
import shutil
import base64
import uuid
import cv2
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image, UnidentifiedImageError

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
templates = Jinja2Templates(directory="templates")

# Model Loading
model = tf.keras.models.load_model("model/best_model_mobileNetV2.keras")
# model = tf.keras.models.load_model("model/fire_today_2.keras")


UPLOAD_DIR = "/tmp/uploads" if os.environ.get("VERCEL") else "static/uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
FACE_DETECTOR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_DETECTOR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")


def allowed_file(filename: str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_data_url(image_path: str, filename: str) -> str:
    extension = filename.rsplit(".", 1)[1].lower()
    mime_type = "image/png" if extension == "png" else "image/jpeg"

    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("ascii")

    return f"data:{mime_type};base64,{encoded}"


def is_eye_area_image(image_path: str) -> bool:
    image = cv2.imread(image_path)
    if image is None:
        return False

    original_height, original_width = image.shape[:2]
    max_side = max(original_width, original_height)
    scale = 640 / max_side if max_side > 640 else 1
    detection_image = image

    if scale != 1:
        detection_image = cv2.resize(
            image,
            (int(original_width * scale), int(original_height * scale))
        )

    gray = cv2.cvtColor(detection_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    detect_height, detect_width = gray.shape[:2]
    detect_area = detect_width * detect_height

    faces = FACE_DETECTOR.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(45, 45)
    )
    eyes = EYE_DETECTOR.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(18, 18)
    )

    max_face_ratio = max(
        ((width * height) / detect_area for _, _, width, height in faces),
        default=0
    )
    max_eye_ratio = max(
        ((width * height) / detect_area for _, _, width, height in eyes),
        default=0
    )

    resized = cv2.resize(image, (224, 224))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    skin_ratio = (
        (((hue < 25) | (hue > 165)) & (saturation > 25) & (saturation < 205) & (value > 45))
    ).mean()
    red_pink_ratio = (
        (((hue < 14) | (hue > 165)) & (saturation > 30) & (value > 60))
    ).mean()
    white_eye_ratio = ((saturation < 48) & (value > 140)).mean()
    uniform_background_ratio = ((saturation < 35) & (value > 150)).mean()

    looks_like_portrait = (
        max_face_ratio > 0.015
        and max_eye_ratio < 0.04
        and uniform_background_ratio > 0.35
    )
    has_large_eye_detection = max_eye_ratio >= 0.06
    has_eye_tissue_colors = (
        red_pink_ratio >= 0.045
        and (skin_ratio >= 0.07 or white_eye_ratio >= 0.08)
        and not (uniform_background_ratio > 0.62 and max_eye_ratio < 0.06)
    )

    return not looks_like_portrait and (has_large_eye_detection or has_eye_tissue_colors)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):

    # 🔐 FILE FORMAT VALIDATION
    if not allowed_file(file.filename):
        return templates.TemplateResponse(
            request,
            "error.html",
            {
                "message": "❌ Invalid file format. Please upload JPG, PNG, or JPEG images only."
            }
        )

    extension = file.filename.rsplit(".", 1)[1].lower()
    safe_filename = f"{uuid.uuid4().hex}.{extension}"
    image_path = os.path.join(UPLOAD_DIR, safe_filename)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 🖼 SAFE IMAGE LOADING
        img = Image.open(image_path).convert("RGB")

        if not is_eye_area_image(image_path):
            return templates.TemplateResponse(
                request,
                "error.html",
                {
                    "title": "Gambar Tidak Bisa Terdeteksi",
                    "message": "Mohon maaf, gambar tidak bisa terdeteksi sebagai area mata. Silahkan upload ulang gambar close-up mata."
                }
            )

        img = img.resize((224, 224))

        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = img_array.reshape(1, 224, 224, 3)

    except UnidentifiedImageError:
        return templates.TemplateResponse(
            request,
            "error.html",
            {
                "message": "❌ Uploaded file is not a valid image.<br> Please upload a correct image file."
            }
        )

    # 🔮 MODEL PREDICTION
    prediction = model.predict(img_array)[0][0]

    # labeling
    label = "Anemia" if prediction >= 0.4 else "Tidak Anemia"
    confidence = prediction * 100 if prediction >= 0.4 else (1 - prediction) * 100

    return templates.TemplateResponse(
        request,
            "result.html",
            {
                "label": label,
                "confidence": f"{confidence:.2f}",
                "image": image_to_data_url(image_path, file.filename)
            }
        )


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)
