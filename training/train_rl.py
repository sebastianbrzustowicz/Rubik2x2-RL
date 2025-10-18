import os
from envs.rubik2x2_env import Rubik2x2Env
from agents.dqn_agent import DQNAgent
import mlflow

def train_rl_agent(
    total_steps=200_000,
    reward_mode="sticker_plus_face",
    scramble_min=1,
    scramble_max=8,
    device="cuda",
    use_mlflow=False
):
    if use_mlflow:
        mlflow.set_experiment("Rubik2x2_DQN")
        run = mlflow.start_run()
        mlflow.log_params({
            "total_steps": total_steps,
            "reward_mode": reward_mode,
            "scramble_min": scramble_min,
            "scramble_max": scramble_max,
            "device": device,
        })

    env = Rubik2x2Env(
        max_steps=50,
        reward_mode=reward_mode,
        scramble_min=scramble_min,
        scramble_max=scramble_max,
        scramble_mode="gradual"
    )

    agent = DQNAgent(env, device=device, use_mlflow=use_mlflow)
    agent.train(total_steps=total_steps)

    os.makedirs("models", exist_ok=True)
    model_path = "models/rl_agent.pth"
    agent.save(model_path)

    if use_mlflow:
        mlflow.end_run()
    
    return model_path


if __name__ == "__main__":
    train_rl_agent()
