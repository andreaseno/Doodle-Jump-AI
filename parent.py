from multiprocessing import Process, Queue
from main import Game
import random
import time

NUM_WORKERS = 3

def worker_fn(queue, data):
    random.seed(1)
    game = Game()
    score = game.run()
    queue.put(f'{data} has a score of {score}')

if __name__ == '__main__':
    
    workers = list()
    
    # create queue
    queue = Queue()
    
    results = list()
    
    for idx in range(NUM_WORKERS):
        workers.append(
            Process(target=worker_fn, args=(queue, idx))
        )
        workers[-1].start()
        time.sleep(0.1)
    
    for idx in range(NUM_WORKERS):
        results.append(queue.get())
        workers[idx].join()
        
    print(results)