# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action) #current successor
        newPos = successorGameState.getPacmanPosition() #position on board
        newFood = successorGameState.getFood() #how much food is left after goes to successor
        newGhostStates = successorGameState.getGhostStates() #where ghosts are after making that move
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] 
        #number of moves that ghosts will be scared
        "*** YOUR CODE HERE ***"
        currPos = currentGameState.getPacmanPosition()
        currFood = currentGameState.getFood()
        currGhostStates = currentGameState.getGhostStates()
    
        #ghostDistances = [manhattanDistance(newPos, state.getPosition()) for state in newGhostStates]
        #closestGhost = min(ghostDistances)

        '''
        currPos = currentGameState.getPacmanPosition()
        currfoods = currentGameState.getFood().asList()
        foodDistances = [manhattanDistance(newPos, food) for food in foods]
        nearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)



        potentialScore = successorGameState.getScore() - currentGameState.getScore()
        '''
        
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent & AlphaBetaPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 7)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        #return self.minimax(0, self.depth*gameState.getNumAgents(), gameState)
        
        def minimax(self, agentIndex, depth, gameState):
            if depth == self.depth or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            #if maximizing player
            if agentIndex == 0: #pacman agent
                maxAction = float('-inf')                
                for legal_action in gameState.getLegalActions(agentIndex):
                    succ_state = gameState.generateSuccessor(agentIndex, legal_action)
                    action_eval = minimax(self, 1, depth, succ_state)
                    maxAction = max(maxAction, action_eval)
                return maxAction
            #if minimizing player
            else:
                minAction = float('inf')
                #for legal_action in gameState.getLegalActions(agentIndex):
                nextIndex = agentIndex + 1
                if gameState.getNumAgents() == nextIndex:
                    nextIndex = 0 #pacman's turn
                if nextIndex == 0:
                    depth += 1 #go to the next depth in the game tree             
                for legal_action in gameState.getLegalActions(agentIndex):
                    succ_state = gameState.generateSuccessor(agentIndex, legal_action)
                    action_eval = minimax(self, nextIndex, depth, succ_state)
                    minAction = min(minAction, action_eval)
                return minAction
                 #   minEval = min(minAction, eval)


        #if the curr_action is BETTER than best_action
        #then set best to current
        #return best move
        maximum_util = float('-inf') #initial value, want to max utility
        bestAction = Directions.WEST #initial value
        for legal_action in gameState.getLegalActions(0):
            succ_state = gameState.generateSuccessor(0, legal_action)
            utility = minimax(self, 1, 0, succ_state)
            if utility > maximum_util or maximum_util == float("-inf"):
                maximum_util = utility
                bestAction = legal_action
        return bestAction  
        '''
        legal_actions = gameState.getLegalActions(0)
        bestMove_value = float('-inf')
        bestMove = None

        for action in legal_actions:
            #curr_action_val = miniMax(gameState.generateSuccessor(0, action), 1, 0)
            curr_action_val = None 
            if curr_action_val > bestMove_value:
                bestMove = curr_action
                bestMove_value = curr_action_val
        
        return bestMove
        '''
            
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 8)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(self, agentIndex, depth, gameState):
            if depth == self.depth or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            #if maximizing player
            if agentIndex == 0: #pacman agent
                maxAction = float('-inf')                
                for legal_action in gameState.getLegalActions(agentIndex):
                    succ_state = gameState.generateSuccessor(agentIndex, legal_action)
                    action_eval = expectimax(self, 1, depth, succ_state)
                    maxAction = max(maxAction, action_eval)
                return maxAction
            #if minimizing player
            else:
                minAction = float('inf')
                #for legal_action in gameState.getLegalActions(agentIndex):
                nextIndex = agentIndex + 1
                if gameState.getNumAgents() == nextIndex:
                    nextIndex = 0 #pacman's turn
                if nextIndex == 0:
                    depth += 1 #go to the next depth in the game tree    
                sum = 0  
                numLegalActions = float(len(gameState.getLegalActions(agentIndex)))       
                for legal_action in gameState.getLegalActions(agentIndex):
                    succ_state = gameState.generateSuccessor(agentIndex, legal_action)
                    action_eval = expectimax(self, nextIndex, depth, succ_state) / numLegalActions
                    sum += action_eval
                    #minAction = min(minAction, action_eval)
                return sum
            
        maximum_util = float('-inf') #initial value
        bestAction = Directions.WEST #initial value
        for legal_action in gameState.getLegalActions(0):
            succ_state = gameState.generateSuccessor(0, legal_action)
            utility = expectimax(self, 1, 0, succ_state)
            if utility > maximum_util or maximum_util == float("-inf"):
                maximum_util = utility
                bestAction = legal_action
        return bestAction  

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 9).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

