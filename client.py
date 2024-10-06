import asyncio
import websockets
import logging

logging.basicConfig(level=logging.INFO)

async def test_websocket():
    uri = "ws://localhost:8080/monitor"  
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                logging.info("Conectado ao WebSocket")

                await websocket.send('{"type": "connect"}')
                
                response = await websocket.recv()
                logging.info(f"Recebido: {response}")
                
                while True:
                    try:
                        message = await websocket.recv()
                        logging.info(f"Nova mensagem: {message}")
                    except websockets.ConnectionClosed:
                        logging.warning("Conexão fechada, tentando reconectar...")
                        break  

        except (ConnectionRefusedError, websockets.InvalidURI) as e:
            logging.error(f"Erro de conexão: {e}")
        
        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(test_websocket())
