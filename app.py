from fastapi import FastAPI, HTTPException, status, Request,  WebSocket, WebSocketDisconnect, Security
from pydantic import BaseModel, Field 
from datetime import datetime 
from typing import Any, List
from dotenv import load_dotenv 
from motor.motor_asyncio import AsyncIOMotorClient
from tenacity import retry, wait_fixed, stop_after_attempt
import asyncio 
import os

load_dotenv()


client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
db = client["wgaze"]
batch_collection = db.batches
app = FastAPI()

trial_buffer = []
connected_clients: List[WebSocket] = []

BATCH_SIZE = 5

class TrialData(BaseModel):
    name: str 
    age: int
    data: Any  

async def process_batch():
    global trial_buffer 
    batch = trial_buffer.copy()
    trial_buffer = []

    batch_id = f"batch_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    batch_data = {
        "batch_id": batch_id,
        "creation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "batch_data": batch
    }

    #await savedb
    print(batch_data)
    #await notify clients


@app.post("/trial")
async def ingest_data(trial_data: TrialData):
    try:
        trial_buffer.append(trial_data.dict())

        if len(trial_buffer) >= BATCH_SIZE:
            #batch procesing
            print(trial_buffer)

        return {"message": "Data ingested successfully"}


    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@retry(wait = wait_fixed(5), stop = stop_after_attempt(5))
async def save_batch_to_db(batch_data):
    try:
        result = batch_collection.insert_one(batch_data)
        
        print(f"Batch saved successfully with id: {result.inserted_id}")

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.websocket('/monitor')
async def monitor(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    print('Cliente conectado')

    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        print('Cliente desconectado')


async def notify_clients(batch_data):
    summary = {
        "batch_id": batch_data["batch_id"],
        "creation_time": batch_data["creation_time"],
        "batch_size": len(batch_data["batch_data"]),
        "status": "batch completed and saved"
    }

    for client in connected_clients:
        try:
            await client.send_json(summary)
        except:
            connected_clients.remove(client)
            print('Erro enviando dado para o cliente. Desconectando...')

