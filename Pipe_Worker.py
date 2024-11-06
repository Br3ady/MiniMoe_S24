import zmq
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Model_config import Config
from Clients import Client1



def encode(data, flag, dest_id):
     buffer = io.BytesIO()
     torch.save((data, flag, dest_id), buffer)
     buffer.seek(0)
     return buffer.getvalue()


def decode(byte_data):
    buffer = io.BytesIO(byte_data)
    tensor, flag, dest_id = torch.load(buffer)
    return tensor, flag, dest_id


context = zmq.Context()
worker = context.socket(zmq.DEALER)
worker.setsockopt(zmq.IDENTITY, b"1")
worker.connect("tcp://127.0.0.1:5555") ###change to tunnel
config = Config()
Model = Client1(config)
optimizer = optim.Adam(Model.parameters(), lr=0.001)

state_dict = Model.state_dict()
for key, value in state_dict.items():
    print(key, "     ", list(value.shape))
    print("\n", end="")
        
breakpoint()

while True:
    try:
        byte_data = worker.recv_multipart()[0] ### dont technically need multipart send [0] is workaround 
        tensor,flag,dest_id = decode(byte_data)

        if flag == 0:
            out = Model(tensor)
        if flag == 1:
            tensor.backwards()

        if flag == 2:
            pass #shuttle data to back pass
        if flag == 3:
            optimizer.step()
            optimizer.zero_grad()

        if flag == 4: ### TODO for loading checkpoints
            state_dict = tensor
            Model.load_state_dict(state_dict,strict=False)
        message = encode(out,flag,dest_id)
        worker.send_multipart([message])

    except zmq.Again:
        pass