import os
from envs.rubik2x2_env import Rubik2x2Env
from agents.dqn_agent import DQNAgent
import mlflow

def train_rl_agent(
    total_steps=200_000,
    max_steps=30,
    reward_mode="sticker_plus_face",
    scramble_min=1,
    scramble_max=8,
    device="cuda",
    use_mlflow=False,
    batch_size=128,
    lr=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.5,
    epsilon_decay=0.9999,
    resets_per_jump=5000,
):
    if use_mlflow:
        mlflow.set_experiment("Rubik2x2_DQN")
        mlflow.start_run()
        mlflow.log_params({
            "total_steps": total_steps,
            "reward_mode": reward_mode,
            "scramble_min": scramble_min,
            "scramble_max": scramble_max,
            "device": device,
        })

    env = Rubik2x2Env(
        max_steps=max_steps,
        reward_mode=reward_mode,
        scramble_min=scramble_min,
        scramble_max=scramble_max,
        scramble_mode="gradual",
        resets_per_jump=resets_per_jump,
    )

    agent = DQNAgent(
        env, 
        device=device, 
        use_mlflow=use_mlflow, 
        batch_size=batch_size,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay
    )
    agent.train(total_steps=total_steps)

    os.makedirs("models", exist_ok=True)
    model_path = "models/rl_agent.pth"
    agent.save(model_path)

    if use_mlflow:
        mlflow.end_run()
    
    return model_path


if __name__ == "__main__":
    train_rl_agent()
