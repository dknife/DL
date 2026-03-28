"""
으뜸 딥러닝 — 15장 02절
FastAPI 모델 서빙 (의사 코드)
"""

# pip install fastapi uvicorn
from fastapi import FastAPI
import torch

app = FastAPI()
model = torch.load("model.pth")
model.eval()

@app.post("/predict")
async def predict(data: dict):
    x = torch.tensor(data["input"])
    with torch.no_grad():
        output = model(x)
    return {"prediction": output.tolist()}

# Run: uvicorn serve:app --host 0.0.0.0 --port 8000
