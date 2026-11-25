"""Cross a lava cell in a grid world with each termination setting."""
import srl_example_setup
from simple_rl.tasks import GridWorldMDP


def main():
    for terminal in (False, True):
        grid = GridWorldMDP(width=3, height=1, lava_locs=[(2, 1)],
                            is_lava_terminal=terminal)
        grid = GridWorldMDP(**grid.get_parameters())
        reward, state = grid.execute_agent_action("right")
        assert reward == -1.0
        assert state.is_terminal() == terminal

        _, next_state = grid.execute_agent_action("right")
        assert next_state.x == (2 if terminal else 3)


if __name__ == "__main__":
    main()
