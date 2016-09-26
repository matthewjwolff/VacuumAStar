import bisect

# Code taken from github.com/aimacode/aima-python/search.py and util.py

def is_in(elt, seq):
    """Similar to (elt in seq), but compares with 'is', not '=='."""
    return any(x is elt for x in seq)

def memoize(fn, slot=None):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, store results in a dictionary."""
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        def memoized_fn(*args):
            if args not in memoized_fn.cache:
                memoized_fn.cache[args] = fn(*args)
            return memoized_fn.cache[args]

        memoized_fn.cache = {}

    return memoized_fn

class Queue:

    """Queue is an abstract class/interface. There are three types:
        Stack(): A Last In First Out Queue.
        FIFOQueue(): A First In First Out Queue.
        PriorityQueue(order, f): Queue in sorted order (default min-first).
    Each type supports the following methods and functions:
        q.append(item)  -- add an item to the queue
        q.extend(items) -- equivalent to: for item in items: q.append(item)
        q.pop()         -- return the top item from the queue
        len(q)          -- number of items in q (also q.__len())
        item in q       -- does q contain item?
    Note that isinstance(Stack(), Queue) is false, because we implement stacks
    as lists.  If Python ever gets interfaces, Queue will be an interface."""

    def __init__(self):
        raise NotImplementedError

    def extend(self, items):
        for item in items:
            self.append(item)

class PriorityQueue(Queue):

    """A queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first. If order is min, the item with minimum f(x) is
    returned first; if order is max, then it is the item with maximum f(x).
    Also supports dict-like lookup."""

    def __init__(self, order=min, f=lambda x: x):
        self.A = []
        self.order = order
        self.f = f

    def append(self, item):
        bisect.insort(self.A, (self.f(item), item))

    def __len__(self):
        return len(self.A)

    def pop(self):
        if self.order == min:
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]

    def __contains__(self, item):
        return any(item == pair[1] for pair in self.A)

    def __getitem__(self, key):
        for _, item in self.A:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.A):
            if item == key:
                self.A.pop(i)

class Node:

    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Create a search tree Node, derived from a parent by an action."
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        "List the nodes reachable in one step from this node."
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        "[Figure 3.10]"
        next = problem.result(self.state, action)
        return Node(next, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next))

    def solution(self):
        "Return the sequence of actions to go from the root to this node."
        return [node.action for node in self.path()[1:]]

    def path(self):
        "Return a list of nodes forming the path from the root to this node."
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

# ______________________________________________________________________________

def best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = PriorityQueue(min, f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
    return None

def astar_search(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))

class Problem(object):

    """The abstract class for a formal problem.  You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError

# Subclass of problem specific to this homework

class VacuumWorld(Problem):
    # A state is a 10-element array consisting of (elements 0-8) the state of each of the rooms and (element 9) the location of the robot
    def actions(self, state):
        actions = ["Suck"]
        # If we're not in square 1, 2 or 3
        if (state[9] > 3):
            actions += ["Up"]
        if (state[9] % 3 != 1):
            actions += ["Left"]
        if (state[9] % 3 != 0):
            actions += ["Right"]
        if (state[9] < 7):
            actions += ["Down"]
        return actions
    
    def goal_test(self, state):
        # Cut off state of robot
        sublist = state[:9]
        return sublist == ("Clean", "Clean", "Clean", "Clean", "Clean", "Clean", "Clean", "Clean", "Clean")
    
    def result(self, state, action):
        # change state (a tuple) to a list to make it mutable
        mutable_state = list(state)
        if action == "Suck":
            mutable_state[mutable_state[9]-1] = "Clean"
        elif action == "Up":
            mutable_state[9] -= 3
        elif action == "Down":
            mutable_state[9] += 3
        elif action == "Left":
            mutable_state[9] -= 1
        elif action == "Right":
            mutable_state[9] += 1
        # now change it back to a tuple to allow for using it as a key
        return tuple(mutable_state)
    
    def path_cost(self, c, state1, action, state2):
        cost = 1
        # For all squares excluding the last one (containing the location of the robot)
        for element in state2[:9]:
            if element=="Dirty":
                cost+=2
        return cost

# create the problem
init_State = ("Dirty", "Dirty", "Dirty", "Clean", "Clean", "Clean", "Clean", "Clean", "Clean", 8)

problem = VacuumWorld(init_State)

def manhattan(a, b):
    # I really couldn't think of a more intelligent way of doing this
    if a==1:
        distances = [0,1,2,1,2,3,2,3,4]
        return distances[b]
    if a==2:
        distances = [1,0,1,2,1,2,3,2,3]
        return distances[b]
    if a==3:
        distances = [2,1,0,3,2,1,4,3,2]
        return distances[b]
    if a==4:
        distances = [1,2,3,0,1,2,1,2,3]
        return distances[b]
    if a==5:
        distances = [2,1,2,1,0,1,2,1,2]
        return distances[b]
    if a==6:
        distances = [3,2,1,2,1,0,3,2,1]
        return distances[b]
    if a==7:
        distances = [2,3,4,1,2,3,0,1,2]
        return distances[b]
    if a==8:
        distances = [3,2,3,2,1,2,1,0,1]
        return distances[b]
    if a==9:
        distances = [4,3,2,3,2,1,2,1,0]
        return distances[b]

def num_to_closest_dirty(state):
    distances = []
    for i in range(len(state[:9])):
        if state[i] == "Dirty":
            distances += [manhattan(state[9], i)]
    if not distances:
        return 0
    return min(distances)
    
# create heuristic function
def heuristic(node):
    num_dirty = 0
    for element in node.state[:9]:
        if element=="Dirty" :
            num_dirty += 1
    dist_to_closest_dirty = num_to_closest_dirty(node.state)
    return num_dirty*dist_to_closest_dirty

# execute A*
result = astar_search(problem, heuristic)
# print required information
# construct list of nodes
path = []
node = result
while node!=None:
    path+=[node]
    node = node.parent
# now reverse order (because its a node->parent->grandparent relationship and we want to go down the family tree)
path.reverse()
print("=====Path=====")
for action in path[1:]:
    print(action.action)
print("=====Listing States=====")
for i in path:
    print (i.state)
print("=====Total Cost=====")
pathCost = 0
for i in path:
    pathCost += i.path_cost
print (pathCost)
