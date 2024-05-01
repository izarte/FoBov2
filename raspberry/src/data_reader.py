#!/usr/bin/env python3
from sensor_msgs.msg import Image
import time
import asyncio
import websockets
import json


class DataReader:
    def __init__(self, port: int, label: str):
        start_server = websockets.serve(self.handler, "0.0.0.0", port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
        self.label = label
        self.data = []
        self.readen = 0

    async def handler(self, websocket):
        data = await websocket.recv()
        # print(data)
        data = json.loads(data)
        self.data = data[self.label]

    def read_data(self) -> any:
        while self.readen:
            if self.readen == 0:
                self.readen = 1
                return self.data
