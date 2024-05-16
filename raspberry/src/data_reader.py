import asyncio
import websockets
import json
import threading


class DataReader:
    def __init__(self, port: int, label: str):
        self.port = port
        self.label = label
        self.data = []
        self.readed = 1
        self.loop = asyncio.new_event_loop()

        server_thread = threading.Thread(target=self.run_server, daemon=True)
        server_thread.start()

    def run_server(self):
        asyncio.set_event_loop(self.loop)
        ready = False
        while not ready:
            try:
                start_server = websockets.serve(self.handler, "0.0.0.0", self.port)
                ready = True
            except:
                continue
        self.loop.run_until_complete(start_server)
        self.loop.run_forever()

    async def handler(self, websocket, path):
        data = await websocket.recv()
        data = json.loads(data)
        self.data = data.get(self.label)
        self.readed = 0

    def read_data(self) -> any:
        while self.readed:
            if self.readed == 0:
                self.readed = 1
                break
        return self.data
