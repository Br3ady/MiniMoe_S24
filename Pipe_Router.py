import zmq
import torch 
import torch.nn as nn
import torch.nn.functional as F
import io
import queue
import threading
import multiprocessing
from test_trainset import OpenWebText
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast
from transformers import GPTNeoForCausalLM
from datasets import load_dataset
from Model_config import Config



def tokenize(dataset): # text to idx 
    return tokenizer(dataset['text'],truncation=True, padding=True)


##to bytes / [tensor, forward/back bool, client_destination]
def encode(data, flag, dest_id):
     buffer = io.BytesIO()
     torch.save((data, flag, dest_id), buffer)
     buffer.seek(0)
     return buffer.getvalue()


def decode(byte_data):
    buffer = io.BytesIO(byte_data)
    buffer.seek(0)
    tensor, flag, dest_id = torch.load(buffer)
    return tensor, flag, dest_id



def forward_thread_func(): # thread 
    print("Forward ON")
    
    while True:
        try:
            data, flag, dest_id = forward_queue.get() #mb, 0, target_client

            message_bytes = encode(data,flag,dest_id)
            dest_bytes = str(dest_id).encode('utf-8')
            router.send_multipart([dest_bytes, message_bytes])
            print(f"sent {data}")
        except queue.Empty:
            pass


def backward_thread_func(): # thread
    print("Backprop ON")
    
    while True:
        try:
            data, flag, dest_id = backward_queue.get() 

            message_bytes = encode(data,flag,dest_id)
            dest_bytes = str(dest_id).encode('utf-8')
            router.send_multipart([dest_bytes, message_bytes])
        except queue.Empty:
            pass


def listen_thread_func(): #thread
    print("Listener ON")
    
    i=0
    j=0
    while True:
        try:
            received_bytes = router.recv_multipart()[0]
            data,flag,dest_id = decode(received_bytes)
            print(flag)
            if flag == 0:
                print("recived 0")
                if dest_id < num_clients: # client_id is 1 idx, recived from client 4 : dest = 4 and needs reversal 
                    dest_id += 1 # i.e. if recived from 2, prep for sending to 3
                    forward_queue.put((data, flag, dest_id)) #keep forwarding
                if dest_id >= num_clients: # after recived from final client
                    target = target_queue.get()
                    loss = Loss_Func(data, target)
                    loss = loss / batch_ratio # scale loss to keep grads consitent 
                    flag = 1 #set to backward flag for next round / dont flip cause we sending back to last client frfr
                    backward_queue.put((loss, flag, dest_id))

            if flag == 1:
                print("Recived 1")
                if dest_id > 1:
                    dest_id -= 1
                    backward_queue.put((data, flag, dest_id))
                if dest_id <= 1:
                    i+=1
                    print(i, end=" ") # mb done

            if flag == 2: # pass shell optim.step token through forward so it lines up 
                if dest_id < num_clients: 
                    dest_id += 1 
                    forward_queue.put((data, flag, dest_id))
                if dest_id >= num_clients: # after recived from final client
                    flag = 3 # set to real optim.step flag
                    backward_queue.put((loss, flag, dest_id))

            if flag == 3: # pass grad update message back after last mb
                if dest_id > 1:
                    dest_id -= 1
                    backward_queue.put((data, flag, dest_id))
                if dest_id <= 1:
                    j+=1
                    next_batch_event.set()
                    print(f"\nBATCH: {j}\n") # end of Batch
        except zmq.Again:
            pass
        
            

def dataloader_thread_func(batch_size,micro_batch_size,dataloader,target_queue,forward_queue,next_batch_event): #  multiprocessing
    print("Dataloader ON")
    i=1
    for microbatch in dataloader:
        i+=1
        if i % (batch_size/micro_batch_size) != 0: 
            init_data = (microbatch["input_ids"],0,1) #set init data (_, forward_flag, to_client_1)
            target_data = (microbatch["labels"])
            target_queue.put(target_data)
            forward_queue.put(init_data)
            print(f"{i} in queue")
        else:
            update_token = (0,2,1) 
            forward_queue.put(update_token)
            next_batch_event.wait()
            next_batch_event.clear()



if __name__ == "__main__":

    multiprocessing.set_start_method("spawn")
    
    num_clients = 2
    batch_size = 2
    micro_batch_size = 1
    batch_ratio = batch_size/micro_batch_size
    assert batch_size % micro_batch_size == 0

    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.bind("tcp://127.0.0.1:5555")
    print("Server Running ... ")

    data = load_dataset("Skylion007/openwebtext") 
    debug_data = data['train'].train_test_split(test_size=0.00001)['test']
    tokenizer = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-neo-1.3B") #(loaded weights specific)
    tokenizer.pad_token = tokenizer.eos_token
    Loss_Func = nn.CrossEntropyLoss()
    config = Config()


    tokenized_data = debug_data.map(tokenize, batched=True, num_proc=8, remove_columns=["text"], batch_size=100) 
    dataset = OpenWebText(config,tokenizer,tokenized_data)
    dataloader = DataLoader(dataset, batch_size=micro_batch_size)



    manager = multiprocessing.Manager() 
    
    forward_queue = manager.Queue()
    backward_queue = manager.Queue()
    target_queue = manager.Queue()
    next_batch_event = manager.Event()

    thread_forward = threading.Thread(target=forward_thread_func, args=())
    thread_backward = threading.Thread(target=backward_thread_func, args=())
    thread_listen = threading.Thread(target=listen_thread_func, args=()) #num_clients,batch_ratio,Loss_Func,target_queue,forward_queue,next_batch_event
    thread_dataloader = multiprocessing.Process(target=dataloader_thread_func, args=(batch_size,micro_batch_size,dataloader,target_queue,forward_queue,next_batch_event))


    print("Running...")
    thread_dataloader.start()
    thread_backward.start()
    thread_listen.start()
    thread_forward.start()


    thread_dataloader.join()
    thread_forward.join()
    thread_listen.join()
    thread_backward.join()