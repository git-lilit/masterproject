import wandb
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from lib.utils import set_seed, initialize_device, compute_auc, TFBindChecker
from lib.dataset_splits import get_saved_dataset, PercentileSplit
from agents.sac.sac_agent import SACAgent
from agents.bc.bc_agent import BCAgent
from agents.dqn.dqn_agent import DQNAgent
from agents.dqn.dqn_agent_ensemble import DQNAgentEnsemble


@hydra.main(version_base=None, config_path="agents/sac", config_name="config")
def main(config: DictConfig):
    wandb_log = True

    # Seed setting
    import torch.backends.cudnn as cudnn

    set_seed(config.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # Initialize all params
    task_info = {"num_actions": 4, "seq_len": 8}

    model_params = {
        "output_dim": task_info["num_actions"],
        "vocab_size": task_info["num_actions"] + 2,
        "seq_len": task_info["seq_len"],
    }

    model_params["embedding_dim"] = config["embedding_dim"]
    model_params["num_heads"] = config["num_heads"]

    train_params_fixed = {
        "training_mode": "offline",
        "n_episodes": 30000,
        "print_interval": 100,
        "esp": None,  # early stopping patience
        "loss_type": "cql",
    }

    train_params = {**task_info, **train_params_fixed, **config}

    X, y = get_saved_dataset()
    full_dataset = (X, y)
    X_train, y_train, X_test, y_test = PercentileSplit(X, y).split(
        lower_percentile=0, upper_percentile=90, sample_fraction=0.2
    )
    train_set = (X_train, y_train)

    # Initialize WandB
    if wandb_log == True:
        seed = config["seed"]
        # job_num = str(HydraConfig.get().job.num)
        # job_num = 0
        wandb.init(
            project="dqn",
            config={**model_params, **train_params},
            name=f"last_sac_run_{seed}",
            group="last_sac",
            reinit=True,
        )
        
    match config["agent_type"]:
        case "bc":
            agent = BCAgent(
                wandb_log=wandb_log,
                device=initialize_device(),
                model_params=model_params,
                test_fn=TFBindChecker(data_folder="data").test_fn,
                dataset=train_set,
            )
        case "dsac":
            agent = SACAgent(
                wandb_log=wandb_log,
                device=initialize_device(),
                model_params=model_params,
                test_fn=TFBindChecker(data_folder="data").test_fn,
                dataset=train_set,
            )
        case "dqn":
            agent = DQNAgent(
                wandb_log=wandb_log,
                device=initialize_device(),
                model_params=model_params,
                test_fn=TFBindChecker(data_folder="data").test_fn,
                dataset=train_set,
                full_dataset=full_dataset
            )
        case "ensemble": 
            agent = DQNAgentEnsemble(
                wandb_log=wandb_log,
                device=initialize_device(),
                model_params=model_params,
                test_fn=TFBindChecker(data_folder="data").test_fn,
                dataset=train_set,
                full_dataset=full_dataset,
                n_networks=5,
                beta=config["beta"],
                integration_type=config["integration_type"]
            )

    all_episode_stats, interval_stats = agent.train(train_params)

    auc = compute_auc(interval_stats)
    last_episodes_rewards = interval_stats["seq_reward"][-5:]
    mean_last_episodes = sum(last_episodes_rewards) / len(last_episodes_rewards)

    final_info = {"auc": auc, "mean_last_episodes": mean_last_episodes}
    if wandb_log == True:
        wandb.log(final_info)

    return auc, mean_last_episodes


if __name__ == "__main__":
    main()
