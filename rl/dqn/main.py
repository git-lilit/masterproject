import wandb
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from lib.TFBindChecker import TFBindChecker
from lib.utils import set_seed, initialize_device, compute_auc
from lib.DQNAgent import DQNAgent
from lib.dataset_splits import get_saved_dataset, RandomSplit


@hydra.main(version_base=None, config_path="", config_name="config")
def main(config: DictConfig):
    wandb_log = False

    # Seed setting
    import torch.backends.cudnn as cudnn

    set_seed(config.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # Initialize all params
    task_info = {"num_states": 4, "seq_len": 8}

    model_params = {
        "embedding_dim": 128,
        "num_heads": 4,
        "output_dim": task_info["num_states"],
        "vocab_size": task_info["num_states"] + 2,
        "seq_len": task_info["seq_len"],
    }

    train_params_fixed = {
        "training_mode": "offline",
        "n_episodes": 10000,
        "print_interval": 100,
        "deterministic": True,
        "esp": None,  # early stopping patience
    }

    train_params = {**task_info, **train_params_fixed, **config}

    # Initialize WandB
    if wandb_log == True:
        # job_num = str(HydraConfig.get().job.num)
        wandb.init(
            project="dqn",
            config={**model_params, **train_params},
            name="with_priority",
            # group="tau_search",
            reinit=True,
        )

    X, y = get_saved_dataset()
    X_train, y_train, X_test, y_test = RandomSplit(X, y).split(train_size=0.5)
    train_set = (X_train, y_train)

    agent = DQNAgent(
        wandb_log=wandb_log,
        device=initialize_device(),
        model_params=model_params,
        test_fn=TFBindChecker(data_folder="data").test_fn,
        dataset=train_set
    )

    all_episode_stats, interval_stats = agent.train(train_params)

    auc = compute_auc(interval_stats)
    last_episodes_rewards = interval_stats['seq_reward'][-10:]
    mean_last_episodes = sum(last_episodes_rewards)/len(last_episodes_rewards)

    return auc, mean_last_episodes


if __name__ == "__main__":
    main()
