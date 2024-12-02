import wandb
import hydra
import torch
import torch.optim as optim
from lib.utils import set_seed
from omegaconf import DictConfig
from lib.ReplayBuffer import ReplayBuffer
from lib.training import train_one_episode, log_and_evaluate, create_fixed_batch
from lib.SequenceModel import SequenceModel
from lib.TFBindChecker import TFBindChecker


@hydra.main(version_base=None, config_path="", config_name="config")
def train(config: DictConfig):
    wandb_log = True
    device = torch.device("cuda:5")
    set_seed(config.seed)

    test_fn_params = {
        "dim": 8,
        "num_states": 4
    }

    checker = TFBindChecker(data_folder="data")
    test_fn = checker.test_fn

    model_params = {
        "embedding_dim": 16,
        "num_heads": 4,
        "output_dim": test_fn_params["num_states"],
        "vocab_size": test_fn_params["num_states"] + 2,
        "seq_len": test_fn_params["dim"]
    }

    train_info = {
        "episodes": [],
        "max_scores": [],
        "epsilons": [],
        "losses": [],
        "td_errors": [],
        "total_rewards_sum": 0,
        "mean_scores": [],
        "mean_losses": [],
        "mean_td_errors": [],
        "mean_top_fives": [],
        "max_qs": [],
    }

    if wandb_log == True:
        wandb.init(
            project="dqn",
            config={**config},
            name="Holo_MC_more_exploration_and_patience_16"
        )

    n_episodes = 2500
    print_interval = 100
    early_stopping_patience = 15

    best_mean_score = float('-inf')
    patience_counter = 0

    # Initialize model
    q = SequenceModel(**model_params, device=device)
    q_target = SequenceModel(**model_params, device=device)

    q_target.load_state_dict(q.state_dict())

    memory = ReplayBuffer(buffer_limit=int(config["buffer_size"]))
    optimizer = optim.Adam(q.parameters(), lr=config["lr"])

    fixed_state_batch = create_fixed_batch(config, test_fn_params, q, device)

    for episode in range(n_episodes):
        epsilon = train_one_episode(q, q_target, memory, optimizer, config, episode,
                                    n_episodes, train_info, test_fn, test_fn_params, device)
        train_info["epsilons"].append(epsilon)

        if episode % print_interval == 0 and episode != 0 and episode > config["learning_starts"]:
            mean_score = log_and_evaluate(q, memory, train_info, config, episode, test_fn, test_fn_params, fixed_state_batch, device)
            
            if mean_score > best_mean_score:
                best_mean_score = mean_score
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered. No improvement in mean score for {early_stopping_patience * print_interval} episodes.")
                break

            episode_stats = {
                "episode": episode,
                "buffer_size": memory.size(),
                "epsilon": epsilon,
                "mean_score": train_info["mean_scores"][-1],
                "max_score": train_info["max_scores"][-1],
                "mean_loss": train_info["mean_losses"][-1],
                "mean_td_errors": train_info["mean_td_errors"][-1],
                "mean_top_5_score": train_info["mean_top_fives"][-1],  # Convert tensor to scalar if necessary
                "max_q": train_info["max_qs"][-1]
            }

            if wandb_log == True:
                wandb.log(episode_stats)

    return best_mean_score


if __name__ == "__main__":
    train()
