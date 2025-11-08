from mouse_cheese import GameBoard
import numpy as np
import os

if __name__ == "__main__":
    pre_trained_states = np.load(os.path.join("pt_states","sarsa_q_obs2.npy"))
    # pre_trained_states = np.load(os.path.join("pt_states","sarsa_q_obs1.npy"))

    attempts = 10
    outcomes = []

    for a in range(attempts):
        game = GameBoard(900, 400, 10, 
                     pre_trained_states, plot_obstacles = True,
                     speed = 10, is_simulation = True)
        
        game.obstacles = game.create_random_obstacles(num = 20)
        game.play()

        outcomes.append(game.game_result)

    sum_outcomes = sum(outcomes)
    print(f"Win: {sum_outcomes} | Loss: {len(outcomes) - sum_outcomes} | Accuracy: {(sum_outcomes / len(outcomes))}")