from agent import Agent
from game import SnakeGameAI
from helper import plot
import argparse
import sys
import os


def train(model_name, headless, reload=None):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(reload)
    game = SnakeGameAI()
    total_number_of_steps = 0
    number_of_steps_in_this_game = 0

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move, slime_grid = agent.get_action(state_old)

        # perform move and get new state

        reward, done, score = game.play_step(final_move, headless, slime_grid)

        state_new = agent.get_state(game)

        # train short memory
        #current_time = time.time()
        #agent.train_short_memory(state_old, final_move, reward, state_new, done)
        #delta_time = time.time() - current_time
        #print(f"Train Short Mem: {delta_time}")

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        total_number_of_steps += 1
        number_of_steps_in_this_game += 1

        if done:
            # train long memory, plot result
            game.reset()
            agent.reset()
            agent.n_games += 1

            if agent.epsilon > 0:
                agent.epsilon -= 1
            # print(f"New epsilon: {agent.epsilon}")

            agent.train_long_memory(number_of_steps_in_this_game)

            if score > record:
                record = score
                agent.model.save(model_name)

            if agent.n_games % 100 == 0:
                agent.model.save('reg_' + model_name)

            total_score += score
            mean_score = total_score / agent.n_games

            plot_scores.append(score)
            plot_mean_scores.append(mean_score)

            sample_count_for_rolling_mean = min(100, agent.n_games)

            scores_for_rolling_mean = plot_scores[-sample_count_for_rolling_mean:]

            rolling_mean = sum(scores_for_rolling_mean) / \
                sample_count_for_rolling_mean

            print(
                f'Game {agent.n_games}, Score {score}, Record: {record}, Average: {mean_score:.2f}, Rolling Average: {rolling_mean:.2f}, Total # of steps: {total_number_of_steps}')

            if not headless:
                plot(plot_scores, plot_mean_scores)

            number_of_steps_in_this_game = 0


def demo(filename, headless=False):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(filename)
    game = SnakeGameAI()
    number_of_steps = 0
    agent.epsilon = -1

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move, slime_grid = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move, headless, slime_grid)

        if done:
            # train long memory, plot result
            game.reset()
            agent.reset()
            agent.n_games += 1

            if score > record:
                record = score

            total_score += score
            mean_score = total_score / agent.n_games

            print(
                f'Game {agent.n_games}, Score {score}, Record: {record}, Average: {mean_score:.1f}')

            if not headless:
                plot_scores.append(score)
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="model.h5",
                        help='Set name of model for autosave.')
    parser.add_argument('--reload', type=str,
                        help='Load a previously saved model.')
    parser.add_argument('--demo', type=str,
                        help='Load up a demo using a saved model.')
    parser.add_argument('--headless', action='store_true',
                        help='Run training without UI (much faster!)')
    args = parser.parse_args(sys.argv[1:])

    if args.demo:
        demo(args.demo, args.headless)
    else:
        train(args.model_name, args.headless, args.reload)


if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    main()
