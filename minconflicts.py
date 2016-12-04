import sys
import heapq
from itertools import count
import random
import time


class PriorityQueue:
    def __init__(self):
        self._queue = []
        self.counter = count()

    def put(self, item, priority):
        heapq.heappush(self._queue, (priority, next(self.counter), item))

    def get(self):
        return heapq.heappop(self._queue)[2]

    def empty(self):
        return len(self._queue) == 0

    def __str__(self):
        return str(self._queue)


class GraphColor:

    def __init__(self, graph, values):

        self.graph = graph
        self.values = values
        self.variables = list(self.graph.keys())
        self.domains = {var: list(self.values) for var in self.variables}

    def isSolution(self, curr):
        for key in curr.keys():
            for neigh in self.graph[key]:
                if curr[key] == curr[neigh]:
                    return False
        return True

    # Randomly selecting the variable from a list of conflicted variables
    def random_select_conflicted(self, curr):
        v=[]
        for key in curr.keys():
            for neigh in self.graph[key]:
                if curr[key] == curr[neigh]:
                    v.append(key)
                    break
        return random.choice(v)

    # Selecting the value which minimizes the conflict
    def minimize_conflict(self, curr, var):
        lis = []
        queue = PriorityQueue()
        for neigh in self.graph[var]:
            lis.append(curr[neigh])
        for val in self.domains[var]:
            c = 0
            for i in lis:
                if val == i:
                    c += 1
            queue.put(val, c)
        return queue.get()

    def minconflict(self):
        curr = {}
        l = len(self.values) - 1
        max_steps = 10000        # Setting the max steps for the algorithm
        for var in self.variables:
            curr[var] = random.randint(0, l)
        for i in range(max_steps):
            if i % 100 == 0:     # Random restart (reassigning state) after every 100 steps
                for var in self.variables:
                    curr[var] = random.randint(0, l)
            if self.isSolution(curr):
                print("No. of Steps: {}".format(i))
                return curr
            var = self.random_select_conflicted(curr)
            val = self.minimize_conflict(curr, var)
            curr[var] = val
        return "fail"

if __name__ == '__main__':

    in_file = open(sys.argv[1], 'r')
    out_file = open(sys.argv[2], 'r+')
    lis = []
    adj = []
    graph = {}
    for line in in_file.readlines():
        lis.append(line.rstrip().split())

    for i in range(0,int(lis[0][0])):
        graph[i] = []

    # Constructing adjacency list from the input
    for edge in lis[1:]:
        graph[int(edge[0])].append(int(edge[1]))
        graph[int(edge[1])].append(int(edge[0]))

    val = range(int(lis[0][2]))
    assignment = {}
    g = GraphColor(graph, val)
    t = time.time()
    asgn = g.minconflict()
    print("Time taken: {} ms".format((time.time() - t)*1000))

    if asgn == "fail":
        print("No Answer - Max Steps Exceeded")
        out_file.write("No Answer")
    else:
        for i in list(asgn.values()):
            out_file.write(str(i) + "\n")