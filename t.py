# from typing import Any, Callable, Iterable, List
# import torch
# from torch.utils.data import Dataset, DataLoader
# from dataclasses import dataclass
# from pathlib import Path

# from torch.utils.data.dataloader import (
#     _BaseDataLoaderIter,
#     _collate_fn_t,
#     _worker_init_fn_t,
# )
# from torch.utils.data.sampler import Sampler


# @dataclass
# class DummyObj:
#     data: torch.Tensor


# class DummyDataset(Dataset):
#     def __init__(self) -> None:
#         super().__init__()
#         self.data = [
#             Path("./data/dummy1.pt"),
#             Path("./data/dummy2.pt"),
#             Path("./data/dummy3.pt"),
#             Path("./data/dummy4.pt"),
#             Path("./data/dummy5.pt"),
#         ]

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         elem = self.data[index]
#         return torch.load(elem, weights_only=True)


# class DummyDataloader(DataLoader):
#     def __init__(
#         self,
#         dataset: Dataset,
#         batch_size: int | None = 1,
#         shuffle: bool | None = None,
#         sampler: Sampler | Iterable | None = None,
#         batch_sampler: Sampler[List] | Iterable[List] | None = None,
#         num_workers: int = 0,
#         # collate_fn: Callable[[List], Any] | None = None,
#         pin_memory: bool = False,
#         drop_last: bool = False,
#         timeout: float = 0,
#         worker_init_fn: Callable[[int], None] | None = None,
#         multiprocessing_context=None,
#         generator=None,
#         *,
#         prefetch_factor: int | None = None,
#         persistent_workers: bool = False,
#         pin_memory_device: str = ""
#     ):
#         super().__init__(
#             dataset,
#             batch_size,
#             shuffle,
#             sampler,
#             batch_sampler,
#             num_workers,
#             self.__collate_fn,
#             pin_memory,
#             drop_last,
#             timeout,
#             worker_init_fn,
#             multiprocessing_context,
#             generator,
#             prefetch_factor=prefetch_factor,
#             persistent_workers=persistent_workers,
#             pin_memory_device=pin_memory_device,
#         )

#     @staticmethod
#     def __collate_fn(batch):
#         print(batch)
#         return batch

#     def __iter__(self):
#         print("nell'iter")

#         yield from super().__iter__()


# # Create dummies if needed
# dataset = DummyDataset()
# for dummy_path in dataset.data:
#     if not dummy_path.exists():
#         torch.save(torch.rand((60 * 1024**3) // 4), dummy_path)
# dataloader = DummyDataloader(dataset, batch_size=1)
# for data in dataloader:
#     print(data)
#     del data


# from functools import partial


# def f(a, b, c):
#     print(a)
#     print(b)
#     print(c)


# partial(partial(partial(f, 1), 2), 3)()

# from pathlib import Path

# p = Path("./models")

# print(p.name)

import numpy as np

action = np.random.normal([0, 0.75, 0.25], [0.5, 0.25, 0.25])

print(
    action.clip(
        -1,
        1,
    )
)
