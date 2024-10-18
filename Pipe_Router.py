import zmq
import torch 
import torch.nn as nn
import torch.nn.functional as F
import time
import io
from test_trainset import RandTensors
from torch.utils.data import DataLoader
from datasets import load_dataset

context = zmq.Context()
router = context.socket(zmq.ROUTER)
router.bind("tcp://127.0.0.1:5555")
dataset = load_dataset("Skylion007/openwebtext")
buffer = io.BytesIO()


def to_bytes(x):
    torch.save(x,buffer)
    buffer.seek(0)
    return buffer.getvalue()


print("Press Enter when all clients connected")
_ = input()
print("Running...")


for i, (A_batch, B_batch, Targets) in enumerate(dataloader):

    A_bytes = to_bytes(A_batch)
    B_bytes = to_bytes(B_batch)
    T_bytes = to_bytes(Targets)
    router.send_multipart([b'worker_1', A_bytes, B_bytes, T_bytes])
    print(f"Sent batch {i}")

    result = router.recv_multipart()
    print(f"recived {int(result[1].decode())}")