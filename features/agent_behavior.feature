Feature: Minesweeper AI Agents
  As a developer
  I want AI agents that can play Minesweeper
  So that I can train and evaluate different strategies

  # ==========================================================================
  # Random Agent
  # ==========================================================================

  Scenario: Random agent selects from valid actions
    Given a new game board
    And a random agent
    When the agent selects an action
    Then the action should be in the valid actions list

  Scenario: Random agent can complete a game
    Given a new beginner game board
    And a random agent
    When the agent plays until game over
    Then the game should be in a terminal state

  # ==========================================================================
  # Logic Agent
  # ==========================================================================

  Scenario: Logic agent identifies safe cells
    Given a board with revealed cell showing 1 adjacent mine
    And only 1 hidden neighbor
    And a logic agent
    When the agent analyzes the board
    Then the agent should flag the hidden neighbor as a mine

  Scenario: Logic agent reveals cells when all mines are flagged
    Given a board with revealed cell showing 1 adjacent mine
    And the adjacent mine is flagged
    And a logic agent
    When the agent selects an action
    Then the agent should reveal the remaining hidden neighbors

  Scenario: Logic agent uses probability when logic is insufficient
    Given an ambiguous board state
    And a logic agent
    When the agent cannot determine safe cells with certainty
    Then the agent should select the cell with lowest mine probability

  # ==========================================================================
  # DQN Agent
  # ==========================================================================

  Scenario: DQN agent receives correct observation shape
    Given a new game board
    And a DQN agent
    When the agent receives an observation
    Then the observation shape should match the board dimensions

  Scenario: DQN agent outputs valid action probabilities
    Given a new game board
    And a trained DQN agent
    When the agent evaluates the board
    Then the output should have probability for each cell

  Scenario: DQN agent learns from experience
    Given a DQN agent with empty replay buffer
    When the agent plays 100 games
    Then the replay buffer should contain experiences
    And the agent's loss should decrease over training

  # ==========================================================================
  # Agent Comparison
  # ==========================================================================

  Scenario: Agents can be benchmarked
    Given a random agent
    And a logic agent
    When each agent plays 100 games
    Then the logic agent should have higher win rate than random
