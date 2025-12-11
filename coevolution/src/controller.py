import random
import sys
import os

# Add the Street_pyghter src directory to the path to import game constants
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../Street_pyghter/src'))

# Import game constants
from Round import KEYCONST, STATECONST


class ActionType:
    """Enumeration of action types the AI can choose."""
    # Movement actions
    FORWARD = 'forward'
    BACKWARD = 'backward'
    JUMP = 'jump'
    CROUCH = 'crouch'

    # Attack actions
    A_ATTACK = 'a_attack'  # Quick attack
    B_ATTACK = 'b_attack'  # Strong attack
    C_ATTACK = 'c_attack'  # Teleport
    THROW = 'throw'  # A+B combination
    SPECIAL = 'special'  # Character-specific special move
    HYPER = 'hyper'  # Hyper combo

    # Defensive actions
    BLOCK = 'block'
    NO_ATTACK = 'no_attack'  # No button press


class GameState:
    """
    Represents the observable game state for decision making.

    This class extracts relevant features from the game objects
    to feed into the AI controller.
    """

    def __init__(self, my_player, opponent, ui=None):
        """
        Initialize game state from player objects.

        Args:
            my_player: Player object (self)
            opponent: Player object (opponent)
            ui: UI object (optional, for timer)
        """
        # Self state
        self.my_position = my_player.position
        self.my_health = my_player.health.hp
        self.my_energy = my_player.energy.energy // 96  # Convert to bars (0-4)
        self.my_state = my_player.getState()
        self.my_facing_right = my_player.facingRight

        # Opponent state
        self.opp_position = opponent.position
        self.opp_health = opponent.health.hp
        self.opp_energy = opponent.energy.energy // 96
        self.opp_state = opponent.getState()
        self.opp_facing_right = opponent.facingRight

        # Derived features
        self.distance = abs(self.my_position.x - self.opp_position.x)
        self.vertical_distance = abs(self.my_position.y - self.opp_position.y)
        self.health_diff = self.my_health - self.opp_health

        # Time remaining (if available)
        self.time_remaining = ui.timer.time if ui else 99

    def is_opponent_attacking(self):
        """Check if opponent is in an attacking state."""
        return self.opp_state == STATECONST.STATE_ATK

    def is_opponent_blocking(self):
        """Check if opponent is blocking."""
        return self.opp_state == STATECONST.STATE_BLCK

    def is_jumping(self):
        """Check if we are jumping."""
        return self.my_state in [STATECONST.STATE_JUMPING, STATECONST.STATE_JUMP]


class AIController:
    """
    AI Controller that converts Individual genotype to game actions.

    Uses a weighted decision tree where different game situations
    trigger different action sets with evolved weights.
    """

    def __init__(self, individual, character_name='Ken'):
        """
        Initialize the AI controller.

        Args:
            individual: Individual object with evolved weights/thresholds
            character_name: Character being controlled (for special moves)
        """
        self.individual = individual
        self.character_name = character_name

        # Extract thresholds from individual
        self.threshold_far = individual.thresholds[0]
        self.threshold_close = individual.thresholds[1]
        self.health_advantage_threshold = individual.thresholds[2]
        self.energy_threshold = individual.thresholds[3]

    def decide_action(self, game_state):
        """
        Decide which action to take based on game state and genotype.

        Args:
            game_state: GameState object

        Returns:
            tuple (stick_inputs, btn_inputs) - lists of KEYCONST values
        """

        # Characters were fighting air at the begginning, without getting close to each other
        if game_state.distance > 200:
            # Force forward movement when too far
            if game_state.my_facing_right:
                return [KEYCONST.FORW], []
            else:
                return [KEYCONST.BACK], []  # left-facing

        # Determine situation and get appropriate weights
        situation = self._categorize_situation(game_state)
        action = self._select_weighted_action(situation)

        # Convert action to inputs
        stick_inputs, btn_inputs = self._action_to_inputs(action, game_state)

        return stick_inputs, btn_inputs

    def _categorize_situation(self, game_state):
        """
        Categorize the current game situation to select weight group.

        Returns:
            Situation index (0-9) corresponding to weight groups
        """
        # Situation 0: FAR + health advantage
        if game_state.distance > self.threshold_far and \
           game_state.health_diff > self.health_advantage_threshold:
            return 0

        # Situation 1: FAR + health disadvantage
        elif game_state.distance > self.threshold_far and \
             game_state.health_diff < -self.health_advantage_threshold:
            return 1

        # Situation 2: CLOSE + has energy
        elif game_state.distance < self.threshold_close and \
             game_state.my_energy >= self.energy_threshold:
            return 2

        # Situation 3: CLOSE + no energy
        elif game_state.distance < self.threshold_close and \
             game_state.my_energy < self.energy_threshold:
            return 3

        # Situation 4: MEDIUM range (default)
        elif self.threshold_close <= game_state.distance <= self.threshold_far:
            return 4

        # Situation 5: JUMPING
        elif game_state.is_jumping():
            return 5

        # Situation 6: OPPONENT attacking
        elif game_state.is_opponent_attacking():
            return 6

        # Situation 7: OPPONENT blocking
        elif game_state.is_opponent_blocking():
            return 7

        # Situation 8: LOW health (<30%)
        elif game_state.my_health < 300:  # Max health is ~1000
            return 8

        # Situation 9: HIGH energy (>=3 bars)
        elif game_state.my_energy >= 3:
            return 9

        # Default to medium range if no specific situation matches
        return 4

    def _select_weighted_action(self, situation):
        """
        Select an action based on weighted probabilities for the situation.

        Args:
            situation: Situation index (0-9)

        Returns:
            ActionType string
        """
        # Get weights for this situation (3 weights per situation)
        weights = self.individual.get_weights_for_situation(situation)
        normalized_weights = self.individual.normalize_weights(weights)

        # Action sets for each situation
        # Made more aggressive to ensure attacks happen and damage is dealt
        action_sets = {
            0: [ActionType.FORWARD, ActionType.A_ATTACK, ActionType.SPECIAL],  # FAR + advantage - approach and attack
            1: [ActionType.FORWARD, ActionType.A_ATTACK, ActionType.BLOCK],  # FAR + disadvantage - approach cautiously
            2: [ActionType.A_ATTACK, ActionType.B_ATTACK, ActionType.THROW],  # CLOSE + energy - aggressive attacks
            3: [ActionType.A_ATTACK, ActionType.A_ATTACK, ActionType.B_ATTACK],  # CLOSE + no energy - fast attacks
            4: [ActionType.A_ATTACK, ActionType.FORWARD, ActionType.B_ATTACK],  # MEDIUM - attack while approaching
            5: [ActionType.A_ATTACK, ActionType.B_ATTACK, ActionType.A_ATTACK],  # JUMPING - aerial attacks
            6: [ActionType.BLOCK, ActionType.A_ATTACK, ActionType.BACKWARD],  # OPP attacking - defensive
            7: [ActionType.THROW, ActionType.A_ATTACK, ActionType.B_ATTACK],  # OPP blocking - break guard
            8: [ActionType.BACKWARD, ActionType.BLOCK, ActionType.A_ATTACK],  # LOW health - survive
            9: [ActionType.SPECIAL, ActionType.B_ATTACK, ActionType.A_ATTACK],  # HIGH energy - use it
        }

        actions = action_sets[situation]

        # Weighted random selection
        action = random.choices(actions, weights=normalized_weights)[0]

        return action

    def _action_to_inputs(self, action, game_state):
        """
        Convert ActionType to game inputs (stick_inputs, btn_inputs).

        Args:
            action: ActionType string
            game_state: GameState object for context

        Returns:
            tuple (stick_inputs, btn_inputs) as lists
        """
        stick_inputs = []
        btn_inputs = []

        # The game environemnt swaps FORW<->BACK for left-facing characters
        # If facing left: BACK to move forward,
        # If facing right: FORW to move forward

        facing_right = game_state.my_facing_right

        # Convert action to inputs
        if action == ActionType.FORWARD:
            if facing_right:
                stick_inputs.append(KEYCONST.FORW)
            else:
                stick_inputs.append(KEYCONST.BACK)  # Pre-swap for left-facing

        elif action == ActionType.BACKWARD:
            if facing_right:
                stick_inputs.append(KEYCONST.BACK)
            else:
                stick_inputs.append(KEYCONST.FORW)  # Pre-swap for left-facing

        elif action == ActionType.JUMP:
            stick_inputs.append(KEYCONST.UP)

        elif action == ActionType.CROUCH:
            stick_inputs.append(KEYCONST.DOWN)

        elif action == ActionType.A_ATTACK:
            btn_inputs.append(KEYCONST.BTNA)

        elif action == ActionType.B_ATTACK:
            btn_inputs.append(KEYCONST.BTNB)

        elif action == ActionType.C_ATTACK:
            btn_inputs.append(KEYCONST.BTNC)

        elif action == ActionType.THROW:
            # Throw is A+B together
            btn_inputs.append(KEYCONST.BTNA)
            btn_inputs.append(KEYCONST.BTNB)

        elif action == ActionType.SPECIAL:
            if facing_right:
                stick_inputs.append(KEYCONST.FORW)
            else:
                stick_inputs.append(KEYCONST.BACK)
            btn_inputs.append(KEYCONST.BTNA)

        elif action == ActionType.HYPER:
            # Simplified hyper: all three buttons
            btn_inputs.append(KEYCONST.BTNA)
            btn_inputs.append(KEYCONST.BTNB)
            btn_inputs.append(KEYCONST.BTNC)

        elif action == ActionType.BLOCK:
            # Block is moving backward
            if facing_right:
                stick_inputs.append(KEYCONST.BACK)
            else:
                stick_inputs.append(KEYCONST.FORW)

        elif action == ActionType.NO_ATTACK:
            # No action - return empty lists
            pass

        return stick_inputs, btn_inputs


# Test the controller
if __name__ == "__main__":
    # Need to import Individual
    from individual import Individual

    # Mock game state for testing
    class MockPosition:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class MockHealth:
        def __init__(self, hp):
            self.hp = hp

    class MockEnergy:
        def __init__(self, energy):
            self.energy = energy

    class MockPlayer:
        def __init__(self, x, y, hp, energy):
            self.position = MockPosition(x, y)
            self.health = MockHealth(hp)
            self.energy = MockEnergy(energy)
            self.facingRight = True

        def getState(self):
            return STATECONST.STATE_IDLE

    # Create test scenario
    print("Testing AIController...")

    # Create an individual
    ind = Individual()
    controller = AIController(ind, 'Ken')

    # Create mock players
    my_player = MockPlayer(100, 195, 800, 192)  # 2 energy bars
    opponent = MockPlayer(250, 195, 600, 96)  # 1 energy bar

    # Create game state
    game_state = GameState(my_player, opponent)

    print(f"\nGame State:")
    print(f"  Distance: {game_state.distance}")
    print(f"  My Health: {game_state.my_health}")
    print(f"  Opp Health: {game_state.opp_health}")
    print(f"  Health Diff: {game_state.health_diff}")
    print(f"  My Energy: {game_state.my_energy} bars")

    # Test decision making
    print(f"\nController Thresholds:")
    print(f"  Far: {controller.threshold_far:.1f}")
    print(f"  Close: {controller.threshold_close:.1f}")
    print(f"  Health Advantage: {controller.health_advantage_threshold:.1f}")
    print(f"  Energy: {controller.energy_threshold:.1f}")

    # Make several decisions
    print(f"\nMaking 5 decisions:")
    for i in range(5):
        stick, btn = controller.decide_action(game_state)
        print(f"  {i+1}. Stick: {stick}, Buttons: {btn}")

    print("\n✓ AIController working correctly!")
