import sys
import heapq
from itertools import count
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
        self.curr_domains = None
        self.search = 0
        self.pruning = 0

    # Checking consistency
    def isConsistent(self, var, color):
        self.load_domain()
        for neigh in self.graph[var]:
            if color == self.curr_domains[neigh]:
                return False
        return True

    def isConsistentPlain(self, var, color, assign):
        for neigh in self.graph[var]:
            if neigh in assign:
                if color == assign[neigh]:
                    return False
        return True

    # Checking constraint satisfaction for arc consistency
    def isconstraint(self, X, x, Y, y):
        return x != y

    def select_unassign_variable(self, assign):
        if len(assign) == 0:
            return self.variables[0]
        for i in self.variables:
            if i not in assign:
                return i

    # Minimum remaining variable
    def select_unassigned_variable(self, assign):
        self.load_domain()
        unassign = [v for v in self.variables if v not in assign]
        min = 9999999
        for key in unassign:
            if min > len(self.curr_domains[key]):
                min = len(self.curr_domains[key])
                v = key
        return v

    # Least constraining values
    def order_domain_values(self, assign, var):
        self.load_domain()
        lis = []
        res = []
        queue = PriorityQueue()
        for neigh in self.graph[var]:
            lis.append(self.curr_domains[neigh])
        k = sum([], lis)
        for val in self.curr_domains[var]:
            c = 0
            for i in k:
                if val == i:
                    c += 1
            queue.put(val, c)
        while not queue.empty():
            res.append(queue.get())
        return res

    def load_domain(self):
        if self.curr_domains is None:
            self.curr_domains = {v: list(self.domains[v]) for v in self.variables}

    # List to maintain the (variable, value) that are removed for a specific variable
    def remove(self, var, value):
        self.load_domain()
        rem = [(var, a) for a in self.curr_domains[var] if a != value]
        self.curr_domains[var] = [value]
        return rem

    # Restore the removal list 
    def restore(self, rem):
        for var, color in rem:
            self.curr_domains[var].append(color)

    # Removing arc inconsistencies
    def remove_inconsistent_values(self, xi, xj, rem):
        removed = False
        for x in self.curr_domains[xi]:
            if all(not self.isconstraint(xi, x, xj, y) for y in self.curr_domains[xj]):
                self.curr_domains[xi].remove(x)
                if rem is not None:
                    rem.append((xi, x))
                removed = True
        return removed

    # Arc consistency code
    def ac3(self, rem):
        queue = [(a, b) for a in self.variables for b in self.graph[a]]
        while len(queue) != 0:
            (xi, xj) = queue.pop()
            if self.remove_inconsistent_values(xi, xj, rem):
                self.pruning += 1
                for xk in self.graph[xi]:
                    queue.append((xk, xi))

    # DSFB code                
    def dsfb_plain(self, assign):
        if len(assign) == len(self.variables):
            return assign
        var = self.select_unassign_variable(assign)
        for color in self.domains[var]:
            if self.isConsistentPlain(var, color, assign):
                assign[var] = color
                res = self.dsfb_plain(assign)
                if res != "fail":
                    return res
                del assign[var]
        return "fail"

    # DSFB++ code
    def dsfb_improved(self, assign):
        self.search += 1
        if len(assign) == len(self.variables):
            print("Search calls are {}".format(self.search))
            print("Arc pruning calls are {}".format(self.pruning))
            return assign
        var = self.select_unassigned_variable(assign)
        for color in self.order_domain_values(assign, var):
            if self.isConsistent(var, color):
                assign[var] = color
                rem = self.remove(var, color)
                self.ac3(rem)
                res = self.dsfb_improved(assign)
                if res != "fail":
                    return res
                self.restore(rem)
                del assign[var]
        return "fail"

if __name__ == '__main__':

    in_file = open(sys.argv[1], 'r')
    out_file = open(sys.argv[2], 'w')
    mode = sys.argv[3]
    lis = []
    adj = []
    graph = {}
    for line in in_file.readlines():
        lis.append(line.rstrip().split())

    for i in range(0,int(lis[0][0])):
        graph[i] = []

    # Constructing adjacency list of the given input
    for edge in lis[1:]:
        graph[int(edge[0])].append(int(edge[1]))
        graph[int(edge[1])].append(int(edge[0]))

    val = range(int(lis[0][2]))
    assignment = {}
    g = GraphColor(graph, val)
    if mode == '0':
        t = time.time()
        asgn = g.dsfb_plain(assignment)
        print("Time taken: {} ms".format((time.time() - t)*1000))
    else:
        t = time.time()
        asgn = g.dsfb_improved(assignment)
        print("Time taken: {} ms".format((time.time() - t)*1000))

    if asgn == "fail":
        print("No Answer")
        out_file.write("No Answer")
    else:
        for i in list(asgn.values()):
            out_file.write(str(i) + "\n")
