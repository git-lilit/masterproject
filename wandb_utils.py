import wandb
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats
from statistics import mean, stdev

# Initialize the API
api = wandb.Api()
entity = "alu-lilit"
project = "dqn"
runs = api.runs(f"{entity}/{project}")


def perform_t_test(group_1, group_2):
    # t_stat, p_value = stats.ttest_ind(group_1, group_2, equal_var=False)

    t_stat, p_value = stats.ttest_rel(group_1, group_2)

    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = "ns"

    return p_value, t_stat, significance


def find_best_run(group_name, n_included_runs, n_finish):
    # Fetch all runs in the project
    runs = api.runs(f"{entity}/{project}")

    # Filter runs by group name
    filtered_runs = [
        run for run in runs if run.group == group_name if run.state == "finished"
    ]

    mean_scores = []

    # Print run details and fetch seq_reward at each timestep
    for run in filtered_runs:
        # Access the history data of the run
        history = run.history()
        print(group_name, history.shape[0])
        if n_finish:
            n_start = n_finish - 1 - n_included_runs
            n_finish = n_finish - 1  # Last row is usually nan
            history = history[n_start:n_finish]
            print(n_start)
            print(n_finish)
        else:
            history = history[-n_included_runs - 1 : -1]

        # Store the rewards for analysis
        mean_scores.append(history[:-1]["seq_reward"][-n_included_runs:].mean())

    best_run_id = np.argmax(mean_scores)

    return filtered_runs[best_run_id].name


def calculate_mean_values(
    group_names, n_included_runs, n_finish=None, originality=False, baseline_idx=None
):
    # Fetch all runs in the project
    all_data = []

    for group_name in group_names:
        print(group_name)
        # Filter runs by group name
        filtered_runs = [run for run in runs if run.group == group_name]

        mean_scores = []
        originality_scores = []

        # Fetch seq_reward at each timestep
        for run in filtered_runs:
            history = run.history()

            if n_finish == None:
                n_finish = history.shape[0] - 1
            # Access the history data of the run
            history = history[n_finish - n_included_runs : n_finish]

            if originality:
                originality_scores.append(history["originality_score"].mean())
            mean_scores.append(history["seq_reward"].mean())

        # Store results in a structured format
        data_entry = {
            "Mean Scores": mean_scores,
            "Group Name": group_name,
            "Mean Seq Reward": round(mean(mean_scores), 3),
            "Std Seq Reward": round(stdev(mean_scores), 3) if len(mean_scores) > 1 else None,
        }
        if originality:
            data_entry["Mean Originality Score"] = mean(originality_scores)

        all_data.append(data_entry)
        
    for data in all_data:
        group_1 = all_data[baseline_idx]["Mean Scores"]
        group_2 = data["Mean Scores"]
        data["P value"], data["T stat"], data["Significance"] = perform_t_test(group_1, group_2)

    # Create DataFrame
    df = pd.DataFrame(all_data)
    df = df.drop("Mean Scores", axis=1)
    return df


def calculate_group_scores(group_name, n_included_runs, n_finish=None):
    # Filter runs by group name
    filtered_runs = [run for run in runs if run.group == group_name]
    print(filtered_runs)
    mean_scores = []

    # Print run details and fetch seq_reward at each timestep
    for run in filtered_runs:
        # Access the history data of the run
        history = run.history()
        if n_finish:
            n_start = n_finish - 1 - n_included_runs
            n_finish = n_finish - 1  # Last row is usually nan
            history = history[n_start:n_finish]
        else:
            history = history[-n_included_runs - 1 : -1]

        mean_scores.append(history["seq_reward"].mean())

    return mean_scores