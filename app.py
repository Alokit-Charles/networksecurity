import sys
import os
import io
import asyncio
from typing import AsyncGenerator
import certifi

ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")

import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline

from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from uvicorn import run as app_run

import pandas as pd
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.constants.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME, 
    DATA_INGESTION_DATABASE_NAME,
    FINALIZED_DIR, 
    FINALIZED_MODEL_OBJECT_NAME, 
    FINALIZED_PREPROCESSING_OBJECT_NAME
)
from networksecurity.utils.feature_extraction.feature_extraction import extract_features_from_url

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="./templates")


@app.exception_handler(NetworkSecurityException)
async def network_security_exception_handler(request: Request, exc: NetworkSecurityException):
    logging.error(str(exc))
    return templates.TemplateResponse(
        request=request,
        name="status.html",
        context={
            "state": "error",
            "title": "Something went wrong",
            "message": "The request couldn't be completed. Details are below.",
            "detail": str(exc),
        },
        status_code=500,
    )


# --- REUSABLE HELPER FOR SINGLE URL SCANS ---
async def process_single_url_scan(request: Request, url: str):
    """Extracts features from a single URL and evaluates it against the model."""
    df_features = extract_features_from_url(url)

    model_path = os.path.join(FINALIZED_DIR, FINALIZED_MODEL_OBJECT_NAME)
    preprocessor_path = os.path.join(FINALIZED_DIR, FINALIZED_PREPROCESSING_OBJECT_NAME)

    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        return templates.TemplateResponse(
            request=request,
            name="status.html",
            context={
                "state": "error",
                "title": "Model Artifacts Not Found",
                "message": "No trained model was found on the server.",
                "detail": f"Missing files in directory '{FINALIZED_DIR}'. Please run model training via the /train route first."
            },
            status_code=404
        )

    preprocessor = load_object(preprocessor_path)
    final_model = load_object(model_path)
    network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

    y_pred = network_model.predict(df_features)
    pred_val = int(round(float(y_pred[0])))

    is_safe = (pred_val == 1)

    return templates.TemplateResponse(
        request=request,
        name="result_single.html",
        context={
            "scanned_url": url,
            "is_safe": is_safe,
            "verdict": "Legitimate" if is_safe else "Phishing",
            "extracted_features": df_features.to_dict(orient="records")[0]
        }
    )


# --- HOMEPAGE ROUTES ---
@app.get("/", tags=["authentication"])
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.post("/")
async def index_url_scan(request: Request, url: str = Form(...)):
    return await process_single_url_scan(request, url)


# --- TRAINING ROUTES ---
@app.get("/train")
async def train_page(request: Request):
    """Renders the training page with progress monitoring UI."""
    return templates.TemplateResponse(request=request, name="train.html", context={})


async def stream_training_progress() -> AsyncGenerator[str, None]:
    """Streams training stages to the browser via SSE."""
    stages = [
        ("10%", "Initiating Data Ingestion from MongoDB..."),
        ("35%", "Validating schema & Checking feature drift..."),
        ("60%", "Transforming dataset & Imputing missing values..."),
        ("85%", "Fitting candidate ML models & Evaluating performance..."),
        ("100%", "Model successfully trained and saved!")
    ]
    
    try:
        train_pipeline = TrainingPipeline()
        
        for progress, msg in stages[:-1]:
            yield f"data: {{\"progress\": \"{progress}\", \"message\": \"{msg}\", \"status\": \"running\"}}\n\n"
            await asyncio.sleep(1.5)
        
        await run_in_threadpool(train_pipeline.run_pipeline)
        
        final_progress, final_msg = stages[-1]
        yield f"data: {{\"progress\": \"{final_progress}\", \"message\": \"{final_msg}\", \"status\": \"completed\"}}\n\n"

    except Exception as e:
        err_msg = str(e).replace('"', '\\"')
        yield f"data: {{\"progress\": \"0%\", \"message\": \"Training Failed: {err_msg}\", \"status\": \"error\"}}\n\n"


@app.get("/train-stream")
async def train_stream():
    return StreamingResponse(stream_training_progress(), media_type="text/event-stream")


# --- BATCH PREDICTION ROUTES ---
@app.get("/predict")
async def predict_form(request: Request):
    return templates.TemplateResponse(request=request, name="predict.html", context={})


@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        model_path = os.path.join(FINALIZED_DIR, FINALIZED_MODEL_OBJECT_NAME)
        preprocessor_path = os.path.join(FINALIZED_DIR, FINALIZED_PREPROCESSING_OBJECT_NAME)

        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            return templates.TemplateResponse(
                request=request,
                name="status.html",
                context={
                    "state": "error",
                    "title": "Model Artifacts Not Found",
                    "message": "No trained model was found on the server.",
                    "detail": f"Missing files in directory '{FINALIZED_DIR}'. Please run model training via the /train route before making batch predictions."
                },
                status_code=404
            )

        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        preprocessor = load_object(preprocessor_path)
        final_model = load_object(model_path)
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        if 'Result' in df.columns:
            df = df.drop(columns=['Result'])

        y_pred = network_model.predict(df)
        df['predicted_column'] = [int(round(float(val))) for val in y_pred]

        os.makedirs("prediction_output", exist_ok=True)
        df.to_csv("prediction_output/output.csv", index=False)

        safe_count = int((df['predicted_column'] == 1).sum())
        phish_count = int((df['predicted_column'] != 1).sum())

        feature_columns = [col for col in df.columns if col != 'predicted_column']

        return templates.TemplateResponse(
            request=request,
            name="result.html",
            context={
                "columns": feature_columns,
                "rows": df.to_dict(orient="records"),
                "row_count": len(df),
                "safe_count": safe_count,
                "phish_count": phish_count,
            },
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys)


@app.post("/predict-url")
async def predict_url_route(request: Request, url: str = Form(...)):
    return await process_single_url_scan(request, url)


if __name__ == "__main__":
    app_run(app, host='0.0.0.0', port=8080)