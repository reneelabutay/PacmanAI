# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys
import copy

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def goalTest(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
        Given a state, returns available actions.
        Returns a list of actions
        """        
        util.raiseNotDefined()

    def getResult(self, state, action):
        """
        Given a state and an action, returns resulting state.
        """
        util.raiseNotDefined()

    def getCost(self, state, action):
        """
        Given a state and an action, returns step cost, which is the incremental cost 
        of moving to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class Node:
    """
    Search node object for your convenience.

    This object uses the state of the node to compare equality and for its hash function,
    so you can use it in things like sets and priority queues if you want those structures
    to use the state for comparison.

    Example usage:
    >>> S = Node("Start", None, None, 0)
    >>> A1 = Node("A", S, "Up", 4)
    >>> B1 = Node("B", S, "Down", 3)
    >>> B2 = Node("B", A1, "Left", 6)
    >>> B1 == B2
    True
    >>> A1 == B2
    False
    >>> node_list1 = [B1, B2]
    >>> B1 in node_list1
    True
    >>> A1 in node_list1
    False
    """
    def __init__(self, state, parent, action, path_cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return self.state == other.state

    def __ne__(self, other):
        return self.state != other.state


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.

    You are not required to implement this, but you may find it useful for Q5.
    """
    
    queue = util.Queue() #empty queue
    visited = []
    queue.push(problem.getStartState())
    path = {}
    
    while queue.isEmpty() == 0:
        node = queue.pop()
        visited.append(node)
        if problem.goalTest(node): #if this is the goal state
            return path[node]
        actions = problem.getActions(node)
        for a in actions:
            succ = problem.getResult(node, a)
            if succ not in visited and succ not in queue.list:
                queue.push(succ)
                path[succ] = path[node] + [a]


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def iterativeDeepeningSearch(problem):
    """
    Perform DFS with increasingly larger depth. Begin with a depth of 1 and increment depth by 1 at every step.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.goalTest(problem.getStartState()))
    print("Actions from start state:", problem.getActions(problem.getStartState()))

    Then try to print the resulting state for one of those actions
    by calling problem.getResult(problem.getStartState(), one_of_the_actions)
    or the resulting cost for one of these actions
    by calling problem.getCost(problem.getStartState(), one_of_the_actions)
    """
    #implement DFS at each depth
    maxDepth = 1 
    while True:
        visited = [] #list of visited nodes
        solution = util.Queue() #list of solutions to print
        stack = util.Stack() #stack for the successors
        result = DLS(problem.getStartState(), problem, maxDepth, solution, visited, stack)
        maxDepth += 1
        if result == True:
            print(result)
            return solution.list
        
        
    

def DLS(node, problem, maxDepth, solution, visited, stack):
    visited.append(node)
    if problem.goalTest(node):
        return True
    elif (maxDepth <= 0): 
        return False
    else:
        x = False
        actions = util.Queue()
        #analyze list of all possible actions at the node
        #add all successors into the stack
        for a in problem.getActions(node): 
            # at node and take action a to get to succ
            succ = problem.getResult(node, a) 
            stack.push(succ)
            actions.push(a)
        #look at the successors in each of the actions
        for a in actions.list:
            succ = stack.pop()
            if visited.count(succ) == 0 and stack.list.count(succ) == 0:
                result = DLS(succ, problem, maxDepth-1, solution, visited, stack)
                if result == 0: #when depth is 0
                    x = True
                elif result == 1: #depth is not 0 yet
                    solution.push(a)
                    return True
        if x: 
            return 0
        else: 
            return None


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    node = problem.getStartState()
    q = util.PriorityQueue() #will keep track of the current node + the current cost
    q.push(node, 0)
    visited = {} #key: node, value: cost 
    paths = {} #key: node, value: path of the node -> will return this path to print 
    paths[node] = [] 
    visited[node] = 0
    while q.isEmpty() == 0: #keep iterating until it's empty
        node = q.pop()
        if problem.goalTest(node): 
            return paths[node] 
        action_list = problem.getActions(node) #has all the possible actions
        for action in action_list: 
            succ = problem.getResult(node, action) #node's successor
            succpath_cost = problem.getCostOfActions(paths[node] + [action])  
            succ_cost = succpath_cost + heuristic(succ, problem) #cost + heuristic of the succ
            if succ not in visited.keys() or visited[succ] > succ_cost:
                paths[succ] = paths[node] + [action] 
                visited[succ] = succ_cost 
                q.push(succ, succ_cost) 
    return None 
    

# Abbreviations
bfs = breadthFirstSearch
astar = aStarSearch
ids = iterativeDeepeningSearch
