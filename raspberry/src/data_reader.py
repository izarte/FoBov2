import asyncio
from websockets import serve
import json
import threading

import numpy as np


class DataReader:
    def __init__(self, port: int, label: str):
        self.port = port
        self.label = label
        self.data = []
        self.readed = 1
        self.loop = asyncio.new_event_loop()

        server_thread = threading.Thread(target=self.run_server)
        server_thread.start()

    def run_server(self):
        asyncio.set_event_loop(self.loop)
        ready = False
        while not ready:
            try:
                start_server = serve(self.handler, "0.0.0.0", self.port)
                ready = True
            except:  # noqa: E722
                continue
        self.loop.run_until_complete(start_server)
        self.loop.run_forever()

    async def handler(self, websocket, path):
        data = await websocket.recv()
        if self.label == "depth":
            self.data = np.frombuffer(data, dtype=np.uint8).reshape(180, 240)
        else:
            data = json.loads(data)
            self.data = data.get(self.label)
        self.readed = 0

    def read_data(self) -> any:
        while self.readed:
            if self.readed == 0:
                self.readed = 1
                break
        return self.data
