import zmq
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import queue
import threading
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


def ping():
    print("pinging server ...")
    ping = encode("1", 0,0)
    worker.send_multipart([ping])
    while True:
        try:
            ping_bytes = worker.recv_multipart()[0]
            print("server response recived")
            break
        except zmq.Again:
            pass


def forward_pass(tensor):
    output = Model(tensor)
    output.ratain_grad()
    forward_queue.put()
    return output

    
def backward_pass(grad, past_output):
    past_output.backward(gradient=grad)
    return past_output.grad


def forward_stream():
    print("Trying forward")
    forward_stream = torch.cuda.Stream()
    with torch.cuda.stream(forward_stream):
        print("Forward started")
        while True:
          tensor = forward_queue.get()
          output = forward_pass(tensor)
          send_queue.put(output)


def backward_stream():
    print("Trying backward")
    backward_stream = torch.cuda.Stream()
    with torch.cuda.stream(backward_stream):
        print("Backward started")
        while True:
          print("looped")
          grad = backward_queue.get()
          past_output = output_queue.get()
          backward_pass(grad,past_output)
    

def port(): # on cpu
    print("Running")
    while True:
        try:
            byte_data = worker.recv_multipart(flags=zmq.NOBLOCK)[0]
            tensor,flag,dest_id = decode(byte_data)

            if flag == 0:
                print("recived0")
                forward_queue.put(tensor)
            
            if flag == 1:
                print("recived1")
                backward_queue.put(tensor)


            if flag == 2:
                print("recived2")
                send_queue.put(tensor)
            if flag == 3:
                print("recived3")
                backward_queue.put(tensor)
            
            out_tensor = send_queue.get()
            print("sending...")
            message = encode(out_tensor,flag,dest_id)
            worker.send_multipart([message])


        except zmq.Again:
            pass    

###############################################################################################################################################
###############################################################################################################################################

context = zmq.Context()
worker = context.socket(zmq.DEALER)
worker.setsockopt(zmq.IDENTITY, b"1")
worker.connect("tcp://2.tcp.ngrok.io:18797") ###change to tunnel
config = Config()
Model = Client1(config)
optimizer = optim.Adam(Model.parameters(), lr=0.001)

forward_queue = queue.Queue()
backward_queue = queue.Queue()
output_queue = queue.Queue()
send_queue = queue.Queue()

port = threading.Thread(target=port)
forward_stream = threading.Thread(target=forward_stream)
backward_stream = threading.Thread(target=backward_stream)


ping()

port.start()
forward_stream.start()
backward_stream.start()



# # Open the browser console (F12), then run:
# function ClickConnect(){
#     document.querySelector("colab-toolbar-button#connect").click()
# }
# setInterval(ClickConnect, 60000);
