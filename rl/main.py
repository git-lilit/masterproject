import wandb
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from lib.utils import set_seed, initialize_device, compute_auc, TFBindChecker
from lib.dataset_splits import get_saved_dataset, PercentileSplit
from agents.bc.bc_agent import BCAgent
from agents.dqn.dqn_agent import DQNAgent
from agents.sac.sac_agent import SACAgent
from agents.dqn.dqn_agent_ensemble import DQNAgentEnsemble
from agents.sac.sac_agent_ensemble import SACAgentEnsemble


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

    if config["with_prior"]:
        model_params["prior_scale"] = config["prior_scale"]

    model_params["embedding_dim"] = config["embedding_dim"]
    model_params["num_heads"] = config["num_heads"]
    loss_type = "cql" if config["agent_type"] == "cql" else "dqn"

    train_params_fixed = {
        "training_mode": "offline",
        "n_episodes": config["n_episodes"],
        "print_interval": 100,
        "esp": None,  # early stopping patience
        "loss_type": loss_type,
    }

    train_params = {**task_info, **train_params_fixed, **config}

    X, y = get_saved_dataset()
    full_dataset = (X, y)
    X_train, y_train, X_test, y_test = PercentileSplit(X, y).split(
        lower_percentile=0,
        upper_percentile=config["upper_percentile"],
        sample_fraction=config["sample_fraction"],
    )
    train_set = (X_train, y_train)
    # Initialize WandB

    if wandb_log == True:
        var_to_include = config["var_to_include"]
        wandb.init(
            project="dqn",
            config={**model_params, **train_params, **config},
            name=f"dsac_run_{config['agent_type']}_{config[var_to_include]:.3f}",
            group=f"{config["group_name"]}",
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
                full_dataset=full_dataset,
            )
        case "cql":
            train_params_fixed["loss_type"] = "cql"
            agent = DQNAgent(
                wandb_log=wandb_log,
                device=initialize_device(),
                model_params=model_params,
                test_fn=TFBindChecker(data_folder="data").test_fn,
                dataset=train_set,
                full_dataset=full_dataset,
            )
        case "ensemble":
            agent = DQNAgentEnsemble(
                wandb_log=wandb_log,
                device=initialize_device(),
                model_params=model_params,
                test_fn=TFBindChecker(data_folder="data").test_fn,
                dataset=train_set,
                full_dataset=full_dataset,
                n_networks=config["n_members"],
                beta=config["beta"],
                integration_type=config["integration_type"],
                bootstrapping=config["bootstrapping"],
                diversification=config["diversification"],
                with_prior=config["with_prior"],
            )
        case "sac_ensemble":
            agent = SACAgentEnsemble(
                wandb_log=wandb_log,
                device=initialize_device(),
                model_params=model_params,
                test_fn=TFBindChecker(data_folder="data").test_fn,
                dataset=train_set,
                n_networks=config["n_members"],
                integration_type=config["integration_type"],
                bootstrapping=config["bootstrapping"],
                diversification=config["diversification"],
                with_prior=config["with_prior"],
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
