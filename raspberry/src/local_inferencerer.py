import asyncio
import websockets
import threading
import json
import numpy as np
import base64


class LocalInferencer:
    def __init__(self, host="0.0.0.0", port=8002):
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = None
        self.server = None
        self.message_recieved = ""
        self.message_available = False
        self.start()

    def start(self):
        # Start the server in a separate thread
        self.thread = threading.Thread(target=self.run_server)
        self.thread.start()

    def run_server(self):
        # Create and set the event loop for this thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Start the WebSocket server
        self.server = websockets.serve(self.handler, self.host, self.port)

        # Execute the server until it completes (it won't under normal conditions)
        self.loop.run_until_complete(self.server)
        self.loop.run_forever()

    def wait_to_client(self):
        import time

        while not self.clients:
            if self.clients:
                print(self.clients)
                time.sleep(3)
                return
            time.sleep(1)

    def read_message(self) -> list:
        while not self.message_available:
            if self.message_available:
                break
        self.message_available = False
        return self.message_recieved["action"]

    async def handler(self, websocket, path):
        self.clients.add(websocket)
        try:
            async for message in websocket:
                self.message_available = True
                self.message_recieved = json.loads(message)
                print(f"Received message from client: {message}")
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
        finally:
            self.clients.remove(websocket)

    def serialize_data(self, data):
        # Convert numpy arrays to base64 encoded strings
        data
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = base64.b64encode(value.tobytes()).decode("utf-8")
        return json.dumps(data)

    def send_message(self, obs: dict):
        message = self.serialize_data(obs)

        # message += obs["wheels_speed"]

        # message = json.dumps(message)
        asyncio.run(self.send_async_message(message))
        # Ensure to use the event loop from the server's thread

        # import time

        # if self.loop:
        #     time.sleep(1)
        #     print("Sending message")
        #     asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)

    async def send_async_message(self, message):
        for websocket in self.clients:
            await websocket.send(message)

    async def broadcast(self, message):
        # Send a message to all connected clients
        if self.clients:
            for client in list(self.clients):
                print("Sending message to", client)
                asyncio.run_coroutine_threadsafe(
                    client.send(message), asyncio.get_event_loop()
                )
                print("Sent for client")

    def stop(self):
        # Stop the server and close the event loop
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()
