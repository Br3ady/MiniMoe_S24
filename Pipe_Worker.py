import zmq
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


context = zmq.Context()
worker = context.socket(zmq.DEALER)
worker.setsockopt_string(zmq.IDENTITY, "worker_1")
worker.connect("tcp://127.0.0.1:5555")


class Custom_Mult(nn.Module):
  def __init__(self, N):
    super(Custom_Mult, self).__init__()
    X = torch.rand(N,N, requires_grad=True)
    self.X = nn.Parameter(X)

  def forward(self, A, B):
    assert A.size()[1] == self.X.size()[0] == B.size()[0]
    mult = A @ self.X @ B
    out = mult.sum()
    return out

model = Custom_Mult(N=250)
optimizer = optim.Adam(model.parameters())
print("Connected...")


while True:
    data_message = worker.recv_multipart()
    A_batch,B_batch,Targets = [torch.load(io.BytesIO(tensor)) for tensor in data_message]
    outs = model(A_batch)
    loss = loss(outs)
    loss.backward()
    optimizer.step()
    

    # worker.send_multipart([str(out).encode()])
    # print("Output Sent")