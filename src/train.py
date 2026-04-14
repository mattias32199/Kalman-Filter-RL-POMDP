from src.environment import PartiallyObservablePendulum
from src.separate_agents import Separate_TD3_EKF_Agent
from src.joint_agents import Joint_TD3_EKF_Agent

def train_joint(
    num_episodes=500,
    max_steps=200,
    batch_size=256,
    explore_noise=0.1,
    warmup_episodes=10,
    noise_std=0.0,
    eval_every=25,
    num_eval_episodes=10,
):
    env = PartiallyObservablePendulum(noise_std=noise_std)
    agent = Joint_TD3_EKF_Agent(max_action=float(env.action_space.high[0])) # Good practice to fetch this from env

    # ── Logging ──
    episode_rewards = []
    eval_rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        agent.reset_ekf(obs)
        ep_reward = 0

        for step in range(max_steps):

            # ── 1. Select action ──
            if ep < warmup_episodes:
                # Uniform random actions for pure exploration during warmup
                action = env.action_space.sample()
            else:
                # Policy action with exploration noise
                action = agent.select_action(obs, explore_noise=explore_noise)

            # ── 2. Environment step ──
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # ── 3. Store transition (Raw observation, NOT EKF state!) ──
            agent.store_transition(obs, action, reward, done)

            # ── 4. Advance EKF with new observation ──
            # (Must happen AFTER storing the current transition, because
            # the buffer pairs obs_t with action_t)
            agent.ekf_step(next_obs, action)

            # ── 5. Train ──
            if ep >= warmup_episodes:
                train_info = agent.train_step(batch_size)

            obs = next_obs
            ep_reward += reward

            if done:
                break

        episode_rewards.append(ep_reward)

        # ── Logging ──
        if (ep + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            Q_mat = agent.ekf.Q.detach().cpu().numpy()
            R_mat = agent.ekf.R.detach().cpu().numpy()
            print(f"Episode {ep+1:4d} | Avg Reward: {avg_reward:8.2f} | "
                  f"Q diag: [{Q_mat[0,0]:.4f}, {Q_mat[1,1]:.4f}] | "
                  f"R diag: [{R_mat[0,0]:.4f}, {R_mat[1,1]:.4f}]")

        # ── Periodic evaluation ──
        if (ep + 1) % eval_every == 0:
            eval_reward = evaluate(env, agent, num_eval_episodes, max_steps)
            eval_rewards.append(eval_reward)
            print(f"  → Eval ({num_eval_episodes} eps): {eval_reward:.2f}")

    return agent, episode_rewards, eval_rewards


def train_separate(
    num_episodes=500,
    max_steps=200,
    batch_size=256,
    explore_noise=0.1,
    warmup_episodes=10,
    noise_std=0.0,
    eval_every=25,
    num_eval_episodes=10,
):
    env = PartiallyObservablePendulum(noise_std=noise_std)
    agent = Separate_TD3_EKF_Agent(max_action=float(env.action_space.high[0]))

    episode_rewards = []
    eval_rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        agent.reset_ekf(obs)
        ep_reward = 0
        ekf_info = {}

        for step in range(max_steps):

            if ep < warmup_episodes:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs, explore_noise=explore_noise)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # CHANGED: extract true_state from info and pass to store_transition
            full = info["full_state"]  # [cos(θ), sin(θ), θ̇]
            true_state = [np.arctan2(full[1], full[0]), full[2]]  # [θ, θ̇]
            agent.store_transition(obs, action, reward, done, true_state)

            agent.ekf_step(next_obs, action)

            # CHANGED: call both training steps
            if ep >= warmup_episodes:
                ekf_info = agent.train_ekf_step(batch_size)
                td3_info = agent.train_step(batch_size)

            obs = next_obs
            ep_reward += reward

            if done:
                break

        episode_rewards.append(ep_reward)

        if (ep + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            Q_mat = agent.ekf.Q.detach().cpu().numpy()
            R_mat = agent.ekf.R.detach().cpu().numpy()
            # CHANGED: also log estimation_loss
            est_loss = ekf_info.get("estimation_loss", float("nan"))
            print(f"Episode {ep+1:4d} | Avg Reward: {avg_reward:8.2f} | "
                  f"Est Loss: {est_loss:.4f} | "
                  f"Q diag: [{Q_mat[0,0]:.4f}, {Q_mat[1,1]:.4f}] | "
                  f"R diag: [{R_mat[0,0]:.4f}, {R_mat[1,1]:.4f}]")

        if (ep + 1) % eval_every == 0:
            eval_reward = evaluate(env, agent, num_eval_episodes, max_steps)
            eval_rewards.append(eval_reward)
            print(f"  → Eval ({num_eval_episodes} eps): {eval_reward:.2f}")

    return agent, episode_rewards, eval_rewards


def evaluate(env, agent, num_episodes, max_steps):
    """Evaluate agent without exploration noise."""
    total_reward = 0
    for _ in range(num_episodes):
        obs, info = env.reset()
        agent.reset_ekf(obs)
        ep_reward = 0

        for step in range(max_steps):
            action = agent.select_action(obs, explore_noise=0.0)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.ekf_step(next_obs, action)
            obs = next_obs
            ep_reward += reward
            if terminated or truncated:
                break

        total_reward += ep_reward
    return total_reward / num_episodes
