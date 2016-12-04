import time
import heapq
from itertools import count
import sys

# Heuristics
# 1) Hamming Distance
# 2) Manhattan distance

# Reference
# 1) https://en.wikipedia.org/wiki/Iterative_deepening_A*
# 2) https://en.wikipedia.org/wiki/A*_search_algorithm
# 3) https://gist.github.com/thiagopnts


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


class Puzzle:
    def __init__(self, initial_state=None):
        self.initial_state = State(initial_state)

    # IDA*
    def idastar_search(self):
        start = self.initial_state
        bound = start.h_man()
        while 1:
            t = self.search(start, 0, bound)
            if t == "goal":
                return bound
            bound = t

    def search(self, state, g, bound):
        f = g + state.h_man()
        if f > bound:
            return f
        if self.is_goal(state):
            out_file.write(', '.join(state.way))
            return "goal"
        min = sys.maxsize
        for child in state.possible_moves():
            child.way = state.way + child.path
            temp = self.search(child, g+1, bound)
            if temp == "goal":
                return "goal"
            if temp < min:
                min = temp
        state.way = ''
        return min


    #goal check
    def is_goal(self, state):
        h = state.h_man()
        if h == 0:
            return True
        else:
           return False

    #hashable
    def hash(self, values):
        return ''.join(str(i) for i in values)

    def construct_path(self, path, state, explored):
        out_file.write(', '.join(state.way))

    # A* Star search
    def astar_search(self):
        frontier = PriorityQueue()
        explored = []
        path = {}
        cost = {}
        start = self.initial_state
        cost[start] = 0
        path[self.hash(start.values)] = ''
        f = 0 + start.h_man()
        frontier.put(start, f)
        while not frontier.empty():
            current = frontier.get()
            if self.is_goal(current):
                return self.construct_path(path, current, explored)
            if current not in explored:
                explored.append(current.values)
            for child in current.possible_moves():
                if child.values in explored:
                    continue
                new_cost = cost[current] + 1
                cost[child] = new_cost
                child.way = current.way + child.path
                f = new_cost + child.h_man()
                frontier.put(child, f)


class State:
    def __init__(self, values, path='', way=None):
        self.values = values
        self.path = path
        self.way = ''
        if flag:
            self.goal = [1,2,3,4,5,6,7,8,0]
        else:
            self.goal = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]

    # Possible moves
    def possible_moves(self):
        i = self.values.index(0)
        if flag:
            if i in [3, 4, 5, 6, 7, 8]:
                puzz_board = self.values[:]
                puzz_board[i], puzz_board[i - 3] = puzz_board[i - 3], puzz_board[i]
                yield State(puzz_board, 'U', self)
            if i in [1, 2, 4, 5, 7, 8]:
                puzz_board = self.values[:]
                puzz_board[i], puzz_board[i - 1] = puzz_board[i - 1], puzz_board[i]
                yield State(puzz_board, 'L', self)
            if i in [0, 1, 3, 4, 6, 7]:
                puzz_board = self.values[:]
                puzz_board[i], puzz_board[i + 1] = puzz_board[i + 1], puzz_board[i]
                yield State(puzz_board, 'R', self)
            if i in [0, 1, 2, 3, 4, 5]:
                puzz_board = self.values[:]
                puzz_board[i], puzz_board[i + 3] = puzz_board[i + 3], puzz_board[i]
                yield State(puzz_board, 'D', self)
        else:
            if i in range(4, 16):
                puzz_board = self.values[:]
                puzz_board[i], puzz_board[i - 4] = puzz_board[i - 4], puzz_board[i]
                yield State(puzz_board, 'U', self)
            if i in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]:
                puzz_board = self.values[:]
                puzz_board[i], puzz_board[i - 1] = puzz_board[i - 1], puzz_board[i]
                yield State(puzz_board, 'L', self)
            if i in [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]:
                puzz_board = self.values[:]
                puzz_board[i], puzz_board[i + 1] = puzz_board[i + 1], puzz_board[i]
                yield State(puzz_board, 'R', self)
            if i in range(0, 12):
                puzz_board = self.values[:]
                puzz_board[i], puzz_board[i + 4] = puzz_board[i + 4], puzz_board[i]
                yield State(puzz_board, 'D', self)


    # hamming distance
    def h(self):
        if flag:
            return sum([1 if self.values[i] != self.goal[i] else 0 for i in [1,2,3,4,5,6,7,8,0]])
        else:
            return sum([1 if self.values[i] != self.goal[i] else 0 for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]])

    #Manhattan distance
    def h_man(self):
        sum = 0
        for i in range(1, n**2):
            sum += self.dist(self.values.index(i), self.goal.index(i))
        return sum

    def dist(self, n, m):
        x1,y1 = coordinates[n]
        x2,y2 = coordinates[m]
        return abs(x1-x2) + abs(y1-y2)



if __name__ == '__main__':

    if len(sys.argv) == 5:
        a = int(sys.argv[1])
        n = int(sys.argv[2])
        in_file = open(sys.argv[3], 'r')
        out_file = open(sys.argv[4], 'r+')
        puzzle = []
        matrix = []
        lis = []
        for line in in_file:
            puzzle.append(line.strip().split(','))

        l = sum(puzzle, [])
        for i in l:
            if i == '':
                matrix.append(0)
            else:
                matrix.append(int(i))
        flag = False
        coordinates = {}
        
        if n == 3:
            flag = True
            coordinates = {0:(0,0), 3:(1,0), 6:(2,0),
                           1:(0,1), 4:(1,1), 7:(2,1),
                           2:(0,2), 5:(1,2), 8:(2,2)}
        elif n == 4:
            coordinates = {0:(0,0), 4:(1,0), 8:(2,0), 12:(3,0),
                           1:(0,1), 5:(1,1), 9:(2,1), 13:(3,1),
                           2:(0,2), 6:(1,2), 10:(2,2), 14:(3,2),
                           3:(0,3), 7:(1,3), 11:(2,3), 15:(3,3)}
        else:
            print("Invalid Matrix size")

        sol = Puzzle(matrix)
        if a == 1:
            sol.astar_search()
        elif a == 2:
            sol.idastar_search()
        else:
            print("Invalid args!")      
    else:
        print('Wrong number of arguments')
