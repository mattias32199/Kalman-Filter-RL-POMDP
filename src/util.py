import json
import os
import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_data(config, seed, group, policy, rewards, evals, agent):
    results = {
        "config": config,
        "seed": seed,
        "agent_type": group,
        "policy_type": policy,
        "episode_rewards": rewards,
        "eval_rewards": evals,
        "final_Q": agent.ekf.Q.detach().cpu().numpy().tolist(),
        "final_R": agent.ekf.R.detach().cpu().numpy().tolist(),
    }

    # save results
    with open(f"results/joint_seed{seed}.json", "w") as f:
        json.dump(results, f, indent=2)

    # save model
    save_dir = f"results/{group}-{policy}-noise{config['noise_std']}-seed{seed}"
    os.makedirs(save_dir, exist_ok=True)

    # Model weights
    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "ekf": agent.ekf.state_dict(),
    }, os.path.join(save_dir, "model.pt"))
