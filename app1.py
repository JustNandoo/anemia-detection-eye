import os
import shutil
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from utils.gradcam import generate_gradcam_pp
from utils.shap_explainer import generate_shap
from PIL import Image, UnidentifiedImageError

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Model Loading
model = tf.keras.models.load_model("model/best_model_mobileNetV2.keras")
# model = tf.keras.models.load_model("model/fire_today_2.keras")


UPLOAD_DIR = "static/uploads"
GRADCAM_DIR = "static/gradcam"
SHAP_DIR = "static/shap"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def allowed_file(filename: str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):

    # 🔐 FILE FORMAT VALIDATION
    if not allowed_file(file.filename):
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "message": "❌ Invalid file format. Please upload JPG, PNG, or JPEG images only."
            }
        )

    image_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 🖼 SAFE IMAGE LOADING
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))

        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = img_array.reshape(1, 224, 224, 3)

    except UnidentifiedImageError:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "message": "❌ Uploaded file is not a valid image.<br> Please upload a correct image file."
            }
        )

    # 🔮 MODEL PREDICTION
    prediction = model.predict(img_array)[0][0]

    # labeling
    label = "Anemic" if prediction >= 0.4 else "Non-Anemic"
    confidence = prediction * 100 if prediction >= 0.4 else (1 - prediction) * 100

    # 🔍 EXPLAINABILITY

    # Grad - CAM + + and SHAP Integration

    gradcam_path = os.path.join(GRADCAM_DIR, "gradcam_" + file.filename)
    shap_path = os.path.join(SHAP_DIR, "shap_" + file.filename)

    generate_gradcam_pp(model, image_path, gradcam_path)
    generate_shap(model, image_path, shap_path)

    return templates.TemplateResponse(
        "result1.html",
        {
            "request": request,
            "label": label,
            "confidence": f"{confidence:.2f}",
            "image": image_path,
            "gradcam": gradcam_path,
            "shap_explainer": shap_path
        }
    )


if __name__ == "__main__":
    uvicorn.run("app1:app", host="127.0.0.1", port=5000, reload=True)
