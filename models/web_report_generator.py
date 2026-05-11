import os
import json
import numpy as np
from openai import OpenAI
from web_rag_utils import retrieve_web_evidence


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def make_json_safe(obj):
    """
    Convert NumPy/Pandas values into normal Python values
    so json.dumps() will not fail.
    """
    if isinstance(obj, dict):
        return {str(key): make_json_safe(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(value) for value in obj]

    if isinstance(obj, tuple):
        return tuple(make_json_safe(value) for value in obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


def generate_web_rag_report(patient_details: dict, features: dict, prediction: dict) -> dict:
    patient_details = make_json_safe(patient_details)
    features = make_json_safe(features)
    prediction = make_json_safe(prediction)

    evidence = retrieve_web_evidence(prediction, top_k=3)
    evidence = make_json_safe(evidence)

    evidence_text = ""

    for i, item in enumerate(evidence, start=1):
        evidence_text += f"""
Source {i}
Title: {item.get("title", "Medical Source")}
URL: {item.get("url", "")}
Snippet: {item.get("snippet", "")}
Content:
{item.get("content", "")}
"""

    prompt = f"""
You are a safe medical explanation assistant.

Patient Details:
{json.dumps(patient_details, indent=2)}

Entered Blood and Morphology Features:
{json.dumps(features, indent=2)}

Screening Prediction:
{json.dumps(prediction, indent=2)}

Trusted Medical Website Evidence:
{evidence_text}

Write a professional RAG-based explanation report.

Required sections:
1. Patient Information
2. Screening Result
3. Important Abnormal Indicators
4. Evidence-Based Explanation
5. Source Support
6. Recommended Clinical Follow-Up
7. Disclaimer

Rules:
- Use only the prediction data and trusted website evidence above.
- Do not invent medical facts.
- Do not prescribe treatment.
- Do not say this is a confirmed diagnosis.
- Say this is a screening result only.
- Mention that confirmation requires professional clinical assessment.
- Include the source names naturally in the report.
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "You generate safe, evidence-grounded medical screening reports. "
                    "You do not provide confirmed diagnosis or treatment."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return {
        "report_text": response.output_text,
        "evidence": evidence,
    }