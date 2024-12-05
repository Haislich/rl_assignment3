from multiprocessing import Process, Queue
import queue
import numpy as np


class Picker:
    def __init__(self) -> None:
        self.max_workers = 100
        self.solutions = Queue(self.max_workers)

    def pick(self, idx):
        while not self.solutions.full():
            elem = np.random.randint(0, 10)
            if elem == 5:
                print("Solution found")
                try:
                    self.solutions.put(idx, block=False)
                except queue.Full:
                    pass
                # SEMAPHORE.release()

    def picker(self):
        processes: list[Process] = []
        for idx in range(self.max_workers):
            process = Process(target=self.pick, args=(idx,))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        for idx in range(self.max_workers):
            print(self.solutions.get(True))
            print(self.solutions.qsize())


Picker().picker()
