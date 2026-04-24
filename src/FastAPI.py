from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Any, Dict
import traceback
import uuid

from src import predict
from src import report_generator

app = FastAPI(title="Blood Cell XAI API with RAG")


class FeatureInput(BaseModel):
    features: Dict[str, Any]


report_store = {}


@app.get("/")
def root():
    return {"message": "Blood Cell XAI API with RAG is running"}


@app.post("/predict")
def predict_endpoint(payload: FeatureInput):
    try:
        result = predict.predict(payload.features)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def build_report(task_id: str, prediction_result: Dict[str, Any]):
    try:
        report_store[task_id]["status"] = "processing"

        rag_result = report_generator.generate_report(prediction_result)

        report_store[task_id]["status"] = "completed"
        report_store[task_id]["report"] = rag_result["report_text"]
        report_store[task_id]["retrieved_context"] = rag_result["retrieved_context"]

    except Exception as e:
        traceback.print_exc()
        report_store[task_id]["status"] = "failed"
        report_store[task_id]["error"] = str(e)


@app.post("/predict-and-report")
def predict_and_report(payload: FeatureInput, background_tasks: BackgroundTasks):
    try:
        prediction_result = predict.predict(payload.features)

        task_id = str(uuid.uuid4())
        report_store[task_id] = {
            "status": "queued",
            "prediction": prediction_result,
            "report": None,
            "retrieved_context": [],
            "error": None
        }

        background_tasks.add_task(build_report, task_id, prediction_result)

        return {
            "message": "Prediction completed. RAG-based report generation started.",
            "task_id": task_id,
            "prediction": prediction_result,
            "report_status": "queued"
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction/report generation failed: {str(e)}")


@app.get("/report/{task_id}")
def get_report(task_id: str):
    if task_id not in report_store:
        raise HTTPException(status_code=404, detail="Task not found")
    return report_store[task_id]