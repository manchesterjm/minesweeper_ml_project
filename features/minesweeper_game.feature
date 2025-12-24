Feature: Minesweeper Game
  As a player
  I want to play Minesweeper
  So that I can enjoy a classic puzzle game

  Background:
    Given a new 9x9 board with 10 mines

  # ==========================================================================
  # Board Initialization
  # ==========================================================================

  Scenario: New board starts with all cells hidden
    Then all cells should be hidden
    And the game state should be "playing"

  Scenario: First click never hits a mine
    When I reveal the cell at row 0, column 0
    Then the game state should be "playing"

  Scenario Outline: Board configurations are validated
    When I create a board with width <width>, height <height>, and <mines> mines
    Then the configuration should be <result>

    Examples:
      | width | height | mines | result  |
      | 9     | 9      | 10    | valid   |
      | 16    | 16     | 40    | valid   |
      | 30    | 16     | 99    | valid   |
      | 0     | 9      | 10    | invalid |
      | 9     | 0      | 10    | invalid |
      | 9     | 9      | 100   | invalid |
      | 9     | 9      | -1    | invalid |

  # ==========================================================================
  # Revealing Cells
  # ==========================================================================

  Scenario: Revealing a safe cell shows adjacent mine count
    Given a board with a mine at row 0, column 0
    When I reveal the cell at row 1, column 1
    Then the cell at row 1, column 1 should show 1 adjacent mine

  Scenario: Revealing an empty cell cascades to neighbors
    Given a board with no mines near row 4, column 4
    When I reveal the cell at row 4, column 4
    Then multiple cells should be revealed

  Scenario: Revealing a mine ends the game
    Given a board with a mine at row 0, column 0
    And mines are already placed
    When I reveal the cell at row 0, column 0
    Then the game state should be "lost"

  Scenario: Cannot reveal an already revealed cell
    When I reveal the cell at row 0, column 0
    And I reveal the cell at row 0, column 0 again
    Then the second reveal should return false

  Scenario: Cannot reveal a flagged cell
    When I flag the cell at row 0, column 0
    And I try to reveal the cell at row 0, column 0
    Then the reveal should return false

  # ==========================================================================
  # Flagging
  # ==========================================================================

  Scenario: Flagging a hidden cell marks it
    When I flag the cell at row 0, column 0
    Then the cell at row 0, column 0 should be flagged

  Scenario: Unflagging a flagged cell returns it to hidden
    When I flag the cell at row 0, column 0
    And I flag the cell at row 0, column 0 again
    Then the cell at row 0, column 0 should be hidden

  Scenario: Cannot flag a revealed cell
    When I reveal the cell at row 0, column 0
    And I try to flag the cell at row 0, column 0
    Then the flag should return false

  # ==========================================================================
  # Winning
  # ==========================================================================

  Scenario: Win by revealing all non-mine cells
    Given a 3x3 board with 1 mine at row 0, column 0
    When I reveal all cells except the mine
    Then the game state should be "won"

  # ==========================================================================
  # Observation for ML
  # ==========================================================================

  Scenario: Board provides observation array for ML agent
    When I request the observation
    Then I should receive a 9x9 numpy array
    And all values should be -1 for hidden cells

  Scenario: Revealed cells show correct values in observation
    When I reveal the cell at row 4, column 4
    And I request the observation
    Then revealed cells should have values 0-8
    And hidden cells should have value -1

  Scenario: Flagged cells show -2 in observation
    When I flag the cell at row 0, column 0
    And I request the observation
    Then the cell at row 0, column 0 should have value -2

  # ==========================================================================
  # Valid Actions
  # ==========================================================================

  Scenario: Get valid actions returns all hidden cells
    Then valid actions should include all 81 cells

  Scenario: Revealed cells are not in valid actions
    When I reveal the cell at row 0, column 0
    Then valid actions should not include row 0, column 0
