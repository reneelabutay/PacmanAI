# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    #Completed
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.q_values = util.Counter()

    #Completed
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.q_values[(state, action)]

    #Completed
    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        '''
        steps:
        - get the legal actions of a state
        - IF 0 legal actions THEN return 0.0
        - ELSE return the MAX ACTION with MAX Q VALUE
        '''
        #max value = maximum q value from a specific action
        #return the VALUE not the action (action will be returned from the function below)
        max_value = float('-inf')
        legal_actions = self.getLegalActions(state)
        if len(legal_actions) == 0:
          return 0.0
        for action in legal_actions:
          curr_value = self.getQValue(state, action)
          if curr_value > max_value:
            max_value = curr_value
        return max_value

    #Completed
    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # return the action with the largest q-val
        legal_actions = self.getLegalActions(state)
        max_action = None
        #No legal actions: at a terminal state so return NONE
        if len(legal_actions) == 0:
          return max_action
        #find the action that has the best q value
        max_value = float('-inf')
        for action in legal_actions:
          curr_value = self.getQValue(state, action)
          if curr_value > max_value:
            max_value = curr_value
            max_action = action
        return max_action

    #Completed
    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        #if there are no legal actions, return None
        if len(legalActions) == 0:
          return action
        #will decide if we will take a random action or not (True or false)
        random_action = util.flipCoin(self.epsilon) 
        if random_action:
          action = random.choice(legalActions) #randomly pick from a list
        else:
          action = self.getPolicy(state) #return the action with the best policy
        return action

    #Completed
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        og_qVal = self.getQValue(state, action) #current q-val
        max_qVal = self.getValue(nextState) #max q-val at the next state
        new_qVal = reward + self.discount * max_qVal
        self.q_values[(state, action)] = og_qVal + self.alpha * (new_qVal - og_qVal)

    #RETURNS the best action to take in a state
    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    #RETURNs max value over legal actions
    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    #Completed
    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        total_qVal = 0
        features = self.featExtractor.getFeatures(state, action)
        for f in features:
          total_qVal += features[f] * self.weights[f]
        return total_qVal

    #Completed
    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        #features for the state and then its action
        features = self.featExtractor.getFeatures(state, action)
        #difference from equation
        difference = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        for f in features:
          self.weights[f] += self.alpha * difference * features[f]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
