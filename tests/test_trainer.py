from concurrent.futures import ProcessPoolExecutor
from time import sleep


def f(a):
    sleep(1)
    return a


with ProcessPoolExecutor(max_workers=1) as executor:
    r = list(executor.map(f, range(30)))
print(r)
