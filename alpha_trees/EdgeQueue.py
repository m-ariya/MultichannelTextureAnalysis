import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class Edge:
    p: int = field(compare=False)
    q: int = field(compare=False)
    alpha: int


class EdgeQueue:
    def __init__(self):
        self.queue = []

    def push(self, edge):
        heapq.heappush(self.queue, edge)

    def pop(self):
        return heapq.heappop(self.queue)

    def empty(self):
        return not self.queue
