from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from datetime import datetime
from typing import Any, List
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from tenacity import retry, wait_fixed, stop_after_attempt
import asyncio
import os
import logging
import json

load_dotenv()

client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
db = client["wgaze"]
batch_collection = db.batches

app = FastAPI()

logging.basicConfig(level=logging.INFO)

trial_buffer: List[dict] = []  
connected_clients: List[WebSocket] = []

BATCH_SIZE = 5  

class TrialData(BaseModel):
    name: str
    age: int
    data: Any

@app.post("/trial")
async def ingest_data(trial_data: TrialData):
    try:
        trial_buffer.append(trial_data.dict())  

        for client in connected_clients:
            await client.send_json(trial_data.dict())  

        result = await batch_collection.insert_one(trial_data.dict())
 

        return {"message": f"Data ingested successfully, id: {result.inserted_id}"}
    except Exception as e:
        # with open('backup.json', 'w') as f:
        #     f.write(trial_buffer)
        logging.error(f"Error ingesting data: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.websocket("/monitor")
async def monitor(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    await websocket.send_json({'message': 'Connected to monitor'})
    logging.info("Client connected")

    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        logging.info("Client disconnected")

