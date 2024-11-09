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
    tensor, flag, dest_id = torch.load(buffer)
    return tensor, flag, dest_id


def forward_thread_func():
    while True:
        try:
            data, flag, dest_id = forward_queue.get() #mb, 0, target_client

            message_bytes = encode(data,flag,dest_id)
            dest_bytes = str(dest_id).encode('utf-8')
            router.send_multipart([dest_bytes, message_bytes])
            print("Sent 0 !!!!!")
        except queue.Empty:
            pass


def backward_thread_func():
    while True:
        try:
            data, flag, dest_id = backward_queue.get() 

            message_bytes = encode(data,flag,dest_id)
            dest_bytes = str(dest_id).encode('utf-8')
            router.send_multipart([dest_bytes, message_bytes])
        except queue.Empty:
            pass


def listen_thread_func():
    i=0
    j=0
    while True:
        received_bytes = router.recv_multipart()
        data,flag,dest_id = decode(received_bytes)

        if flag == 0:
            print("recived 0")
            if dest_id < num_clients: # client_id is 1 idx, recived from client 4 : dest = 4 and needs reversal 
                dest_id += 1 # if recived form 2, prep for sending to 3
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
            

def dataloader_thread_func():
    i=1
    print("loading")
    for microbatch in dataloader:
        i+=1
        if i % (batch_size/micro_batch_size) != 0: 
            init_data = (microbatch["input_ids"],0,1) #set init data (_, forward_flag, to_client_1)
            target_data = (microbatch["labels"])
            target_queue.put(target_data)
            forward_queue.put(init_data)
        else:
            update_token = (0,2,1) 
            forward_queue.put(update_token)
            next_batch_event.wait()
            next_batch_event.clear()


### TODO Not complete  // turn to tractable functions and forward 
### for init loding different model checkpoints # change per model type and ensure layer size match
def load_checpoint(path): # "gpt_model_checkpoint.pth" 
    state_dict = torch.load(path) 
    client_dicts = [{} for _ in range(num_clients)] # for sub_dicts

    ### split into clients
    for name, param in state_dict.items():
        if name.startswith("transformer.h"):

            layer_num = int(name.split(".")[2])  # Get layer number
            client_id = layer_num // config.n_layer  # Determine client id
            client_dicts[client_id][name] = param
        else:
            for client_dict in client_dicts:
                client_dict[name] = param



    ### combine q,k,v proj
    for client_dict in client_dicts: ### each client dict 
        q_large, k_large, v_large = [],[],[] 
        for key,value in client_dict.items(): # add each 
            if key.endswith("attn.attention.q_proj.weight"):
                q_large.append(key)
            


    ### itt and print for comp
    for i,client_dict in enumerate(client_dicts):
        
        for key,value in client_dict.items():
            key=key.replace("transformer.",'')
            print(key, "     ", list(value.shape))
            print("\n", end="")
        
        breakpoint()


        dest_bytes = str(i).encode('utf-8')
        dict_bytes = encode(client_dict,4,i)
        router.send_multipart([dest_bytes,dict_bytes])



if __name__ == "__main__":

    num_clients = 4
    batch_size = 32
    micro_batch_size = 8
    batch_ratio = batch_size/micro_batch_size
    assert batch_size % micro_batch_size == 0

    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.bind("tcp://127.0.0.1:5555")

    data = load_dataset("Skylion007/openwebtext") 
    debug_data = data['train'].train_test_split(test_size=0.001)['test']
    tokenizer = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-neo-1.3B") #(loaded weights specific)
    tokenizer.pad_token = tokenizer.eos_token
    Loss_Func = nn.CrossEntropyLoss()
    config = Config()


    tokenized_data = debug_data.map(tokenize, batched=True, num_proc=8, remove_columns=["text"], batch_size=100) 
    dataset = OpenWebText(config,tokenizer,tokenized_data)
    dataloader = DataLoader(dataset, batch_size=micro_batch_size)

    forward_queue = queue.Queue()
    backward_queue = queue.Queue()
    target_queue = queue.Queue()
    next_batch_event = threading.Event()
    thread_forward = threading.Thread(target=forward_thread_func)
    thread_backward = threading.Thread(target=backward_thread_func)
    thread_listen = threading.Thread(target=listen_thread_func)
    thread_dataloader = threading.Thread(target=dataloader_thread_func)



    print("Press Enter when all clients connected")
    _ = input()
    print("Running...")


    thread_dataloader.start()
    thread_forward.start()
    thread_listen.start()
    thread_backward.start()
    print("threadin")