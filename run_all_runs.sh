/home/ghandill/miniconda3/envs/rl/bin/python /home/ghandill/masterproject_lilit_ghandilyan/rl/main.py \
    -m seed=1\
    agent_type=ensemble \
    integration_type=var \
    group_name=random_runs_correlation_trials;

# /home/ghandill/miniconda3/envs/rl/bin/python /home/ghandill/masterproject_lilit_ghandilyan/rl/main.py \
#     -m seed=1,2,3,4,5,6,7,8,9,10\
#     agent_type=dsac \
#     group_name=random_runs_dsac_seeds;

# /home/ghandill/miniconda3/envs/rl/bin/python /home/ghandill/masterproject_lilit_ghandilyan/rl/main.py \
#     -m seed=1,2,3,4,5,6,7,8,9,10\
#     agent_type=sac_ensemble \
#     integration_type=min \
#     group_name=random_runs_dsac_seeds_min;

# /home/ghandill/miniconda3/envs/rl/bin/python /home/ghandill/masterproject_lilit_ghandilyan/rl/main.py \
#     -m seed=1,2,3,4,5,6,7,8,9,10\
#     agent_type=sac_ensemble \
#     integration_type=var \
#     beta=0.2 \
#     group_name=random_runs_dsac_seeds_var;

# /home/ghandill/miniconda3/envs/rl/bin/python /home/ghandill/masterproject_lilit_ghandilyan/rl/main.py \
#     -m seed=1,2,3,4,5,6,7,8,9,10\
#     agent_type=sac_ensemble \
#     integration_type=var \
#     bootstrapping=True \
#     beta=0.073 \
#     bernoulli_p=0.9 \
#     group_name=random_runs_dsac_seeds_bootstrapping;

# /home/ghandill/miniconda3/envs/rl/bin/python /home/ghandill/masterproject_lilit_ghandilyan/rl/main.py \
#     -m seed=1,2,3,4,5,6,7,8,9,10\
#     agent_type=sac_ensemble \
#     integration_type=var \
#     diversification=True \
#     eta=0.003 \
#     beta=0.084 \
#     group_name=random_runs_dsac_seeds_div;

# /home/ghandill/miniconda3/envs/rl/bin/python /home/ghandill/masterproject_lilit_ghandilyan/rl/main.py \
#     -m seed=1,2,3,4,5,6,7,8,9,10\
#     agent_type=sac_ensemble \
#     integration_type=var \
#     with_prior=True \
#     beta=0.011 \
#     prior_scale=0.154 \
#     group_name=random_runs_dsac_seeds_prior;




# /home/ghandill/miniconda3/envs/rl/bin/python /home/ghandill/masterproject_lilit_ghandilyan/rl/main.py \
#     -m seed=1,2,3,4,5,6,7,8,9,10\
#     agent_type=sac_ensemble \
#     integration_type=var \
#     embedding_dim=32 \
#     num_heads=4 \ 
#     group_name=random_runs_dsac_seeds_small;
