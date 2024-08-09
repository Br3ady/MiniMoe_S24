from tqdm import tqdm
import time
import random

epochs = 1280

for epoch in tqdm(range(epochs)):
    time.sleep(random.randint(5,20))