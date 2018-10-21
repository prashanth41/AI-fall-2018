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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        prevFood = currentGameState.getFood()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        closestGhost = 1000
        for state in newGhostStates:
          distance = manhattanDistance(state.getPosition(),newPos)
          if distance < closestGhost: 
            closestGhost = distance

        numFood = successorGameState.getNumFood()
        previousNumFood = currentGameState.getNumFood()

        if closestGhost <= 1:
          return -1000

        minimumDistanceFood = 0
        distancesFoods = []
        for food in newFood.asList() :
            distancesFoods.append(manhattanDistance(newPos,food))
        if len(distancesFoods) > 0 :
            minimumDistanceFood = min(distancesFoods)
        return -minimumDistanceFood + successorGameState.getScore() + 100*(previousNumFood - numFood)
        

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
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
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
      Your minimax agent (question 2)
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
          gameState.isWin():
            Returns whether or not the game state is a winning state
          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(self.index)
        highest = float("-inf")
        bestAction = None
        maximumDepth = self.depth * gameState.getNumAgents()
        for action in actions:
          state = gameState.generateSuccessor(self.index,action)
          score = value(state,1,maximumDepth,self.evaluationFunction,self.index)
          if score > highest: 
            highest = score
            bestAction = action
        return bestAction
        
def value(gameState,depth,maximumDepth,evaluationFunction,agentIndex):
  if depth == maximumDepth or gameState.isWin() or gameState.isLose():
    return evaluationFunction(gameState)
  nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
  if nextAgentIndex == 0:
    return maximumValue(gameState,depth+1,maximumDepth,evaluationFunction,nextAgentIndex)
  else: 
    return minimumValue(gameState,depth+1,maximumDepth,evaluationFunction,nextAgentIndex)


def maximumValue(gameState,depth,maximumDepth,evaluationFunction,agentIndex):
  maxValue = float("-inf")
  actions = gameState.getLegalActions(agentIndex)
  for action in actions:
    state = gameState.generateSuccessor(agentIndex,action)
    maxValue = max(maxValue,value(state,depth,maximumDepth,evaluationFunction,agentIndex))
  return maxValue

def minimumValue(gameState,depth,maximumDepth,evaluationFunction,agentIndex):
  minValue = float("inf")
  actions = gameState.getLegalActions(agentIndex)
  for action in actions:
    state = gameState.generateSuccessor(agentIndex,action)
    minValue = min(minValue,value(state,depth,maximumDepth,evaluationFunction,agentIndex))
  return minValue

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(self.index)
        highest = float("-inf")
        bestAction = None
        maximumDepth = self.depth * gameState.getNumAgents()
        alpha = float("-inf")
        beta = float("inf")
        for action in actions:          
          state = gameState.generateSuccessor(self.index,action)
          # ALPHA GETS UPDATED BASED ON BEST SCORE FROM PREVIOUS SUCCESSOR. 
          # In this case, it is the 'highest' variable.
          score = value_alphabeta(state,1,maximumDepth,self.evaluationFunction,self.index,highest,beta)
          if score > highest: 
            highest = score
            bestAction = action
        return bestAction

def value_alphabeta(gameState,depth,maximumDepth,evaluationFunction,agentIndex,alpha,beta):
  if depth == maximumDepth or gameState.isWin() or gameState.isLose():
    return evaluationFunction(gameState)
  nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
  if nextAgentIndex == 0:
    return maximumValue_alphabeta(gameState,depth+1,maximumDepth,evaluationFunction,nextAgentIndex,alpha,beta)
  else: 
    return minimumValue_alphabeta(gameState,depth+1,maximumDepth,evaluationFunction,nextAgentIndex,alpha,beta)


def maximumValue_alphabeta(gameState,depth,maximumDepth,evaluationFunction,agentIndex,alpha,beta):
  maxValue = float("-inf")
  actions = gameState.getLegalActions(agentIndex)
  for action in actions:
    state = gameState.generateSuccessor(agentIndex,action)
    maxValue = max(maxValue,value_alphabeta(state,depth,maximumDepth,evaluationFunction,agentIndex,alpha,beta))
    if maxValue > beta: 
      return maxValue
    alpha = max(alpha,maxValue)
  return maxValue

def minimumValue_alphabeta(gameState,depth,maximumDepth,evaluationFunction,agentIndex,alpha,beta):
  minValue = float("inf")
  actions = gameState.getLegalActions(agentIndex)
  for action in actions:
    state = gameState.generateSuccessor(agentIndex,action)
    minValue = min(minValue,value_alphabeta(state,depth,maximumDepth,evaluationFunction,agentIndex,alpha,beta))
    if minValue < alpha: 
      return minValue
    beta = min(beta,minValue)
  return minValue
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(self.index)
        highest = float("-inf")
        bestAction = None
        maximumDepth = self.depth * gameState.getNumAgents()
        for action in actions:
          state = gameState.generateSuccessor(self.index,action)
          score = value_ex(state,1,maximumDepth,self.evaluationFunction,self.index)
          if score >= highest: 
            highest = score
            bestAction = action
        return bestAction

def value_ex(gameState,depth,maximumDepth,evaluationFunction,agentIndex):
  if depth == maximumDepth or gameState.isWin() or gameState.isLose():
    return evaluationFunction(gameState)
  nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
  if nextAgentIndex == 0:
    return maxValue_ex(gameState,depth+1,maximumDepth,evaluationFunction,nextAgentIndex)
  else: 
    return avgValue_ex(gameState,depth+1,maximumDepth,evaluationFunction,nextAgentIndex)


def maxValue_ex(gameState,depth,maximumDepth,evaluationFunction,agentIndex):
  maxValue = float("-inf")
  actions = gameState.getLegalActions(agentIndex)
  for action in actions:
    state = gameState.generateSuccessor(agentIndex,action)
    maxValue = max(maxValue,value_ex(state,depth,maximumDepth,evaluationFunction,agentIndex))
  return maxValue

def avgValue_ex(gameState,depth,maximumDepth,evaluationFunction,agentIndex):
  avgValue = []
  actions = gameState.getLegalActions(agentIndex)
  for action in actions:
    state = gameState.generateSuccessor(agentIndex,action)
    avgValue.append(float(value_ex(state,depth,maximumDepth,evaluationFunction,agentIndex)))

  return sum(avgValue)/float(len(avgValue))

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    prevFood = currentGameState.getFood()
    currScore = currentGameState.getScore
    actions = currentGameState.getLegalActions()

    pos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    numFood = currentGameState.getNumFood()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    closestGhost = 1000
    for state in ghostStates:
      distance = manhattanDistance(state.getPosition(),pos)
      if distance < closestGhost: 
        closestGhost = distance

    if closestGhost <= 1 and scaredTimes == 0:
      return -10000

    minimumDistanceFood = 0
    distancesFoods = []
    for food in prevFood.asList() :
        distancesFoods.append(manhattanDistance(pos,food))
    if len(distancesFoods) > 0 :
        minimumDistanceFood = min(distancesFoods)

    return -minimumDistanceFood + currentGameState.getScore() + 100*(closestGhost <= 1 and scaredTimes > 0)


# Abbreviation
better = betterEvaluationFunction