from typing import Any


class A:
    def __init__(self) -> None:
        a = 0

    def f(self, a):
        self(a)
        print(a)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print("culottide")


class B(A):
    def __init__(self) -> None:
        super().__init__()

    def f(self, a, b):
        super().f(a)
        print(b)


B().f(1, 2)
