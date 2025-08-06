from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi_app.model_utils import predict

app = FastAPI()

# Enable CORS so Streamlit can call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local testing
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_skin_disease(file: UploadFile = File(...)):
    image_bytes = await file.read()
    prediction = predict(image_bytes)
    return {"prediction": prediction}
