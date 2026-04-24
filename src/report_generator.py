import os
import json
from openai import OpenAI
from src.rag_utils import LocalRAGRetriever

# Load OpenAI client safely from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Knowledge base folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "data", "knowledge_base")

# Create retriever once
retriever = LocalRAGRetriever(KNOWLEDGE_DIR)


def build_query_from_prediction(prediction_json: dict) -> str:
    predicted_cell_type = prediction_json.get("predicted_cell_type", "unknown")
    anomaly_label = prediction_json.get("predicted_anomaly_label", "unknown")
    confidence = prediction_json.get("confidence", "unknown")
    anomaly_score = prediction_json.get("anomaly_score", "unknown")

    query = (
        f"Blood cell type: {predicted_cell_type}. "
        f"Anomaly label: {anomaly_label}. "
        f"Confidence: {confidence}. "
        f"Anomaly score: {anomaly_score}. "
        f"Retrieve medical explanation for this blood cell finding."
    )
    return query


def generate_report(prediction_json):
    try:
        # Step 1: Build retrieval query
        retrieval_query = build_query_from_prediction(prediction_json)

        # Step 2: Retrieve top knowledge chunks
        retrieved_chunks = retriever.retrieve(retrieval_query, top_k=3)

        retrieved_context = "\n\n".join(
            [f"Source {i+1}: {chunk}" for i, chunk in enumerate(retrieved_chunks)]
        )

        # Step 3: Prepare grounded prompt
        system_prompt = (
            "You are a medical report generator for blood cell anomaly screening. "
            "Use only the provided prediction data and retrieved medical knowledge. "
            "Do not invent facts. "
            "Do not provide prescriptions or treatment plans. "
            "Only provide educational, general, non-prescriptive explanations. "
            "If uncertainty exists, clearly say manual clinical review is needed."
        )

        user_prompt = f"""
Generate a structured medical report based on:

Prediction Data:
{json.dumps(prediction_json, indent=2)}

Retrieved Medical Knowledge:
{retrieved_context}

Write the report with these sections:
1. Predicted Cell Type
2. Anomaly Interpretation
3. Confidence Analysis
4. Medical Knowledge Support
5. Suggested Next Steps (non-prescriptive)
6. Disclaimer

Keep the language simple, clear, and explainable.
"""

        # Step 4: Call OpenAI
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return {
            "report_text": response.output_text,
            "retrieved_context": retrieved_chunks
        }

    except Exception as e:
        raise Exception(f"OpenAI RAG report generation failed: {str(e)}")