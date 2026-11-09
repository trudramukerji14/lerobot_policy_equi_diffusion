"""Train a LeRobot policy (ported to lerobot 0.7.0).

Examples:
  # quick local smoke test
  uv run python train.py --steps 20 --batch-size 4 --device cpu --overwrite

  # real run on a rented GPU
  uv run python train.py --steps 200000 --batch-size 64 --device cuda --num-workers 8 --trackio
"""
import argparse
import shutil
from pathlib import Path

from lerobot.configs.default import DatasetConfig, TrackioConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.policies.factory import make_policy_config
from lerobot.scripts import lerobot_train


def start_training(
    repo_id="lerobot/pusht",
    root=None,
    output_dir="./outputs",
    policy_name="diffusion",
    job_name="",
    device="cpu",
    steps=10000,
    batch_size=4,
    log_freq=200,
    save_freq=1000,
    env_eval_freq=0,
    num_workers=4,
    trackio=False,
    overwrite_checkpoint=False,
    episodes=None,
    horizon=None,
    n_action_steps=None,
    drop_features=None,
):
    dataset = DatasetConfig(repo_id=repo_id, root=root, episodes=episodes)
    policy = make_policy_config(policy_name)
    policy.push_to_hub = False
    policy.device = device
    if drop_features:
        # Pin the policy's inputs to the dataset features minus the dropped ones
        # (e.g. metaworld's observation.environment_state, which the sim doesn't provide).
        meta = LeRobotDatasetMetadata(repo_id, root=root)
        feats = dataset_to_policy_features(meta.features)
        policy.input_features = {
            k: v for k, v in feats.items() if v.type is not FeatureType.ACTION and k not in drop_features
        }
    if horizon is not None:
        policy.horizon = horizon
    if n_action_steps is not None:
        policy.n_action_steps = n_action_steps

    full_job_name = f"{job_name or 'train'}_{policy_name}_{repo_id.replace('/', '_')}"
    output_dir = Path(output_dir) / full_job_name

    # lerobot_train errors if output_dir is non-empty and resume is False
    if overwrite_checkpoint and output_dir.exists():
        print(f"Removing directory: {output_dir}")
        shutil.rmtree(output_dir)

    train_config = TrainPipelineConfig(
        dataset=dataset,
        policy=policy,
        trackio=TrackioConfig(enable=trackio, project="equi_diffusion", run_name=full_job_name),
        output_dir=output_dir,
        job_name=full_job_name,
        batch_size=batch_size,
        steps=steps,
        log_freq=log_freq,
        save_freq=save_freq,
        env_eval_freq=env_eval_freq,
        num_workers=num_workers,
    )

    lerobot_train.train(train_config)
    return output_dir / "checkpoints" / "last" / "pretrained_model"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="lerobot/pusht")
    p.add_argument("--root", default=None)
    p.add_argument("--output-dir", default="./outputs")
    p.add_argument("--policy", default="diffusion")
    p.add_argument("--job-name", default="")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--log-freq", type=int, default=200)
    p.add_argument("--save-freq", type=int, default=1000)
    p.add_argument("--env-eval-freq", type=int, default=0, help="sim eval every N steps (0 = off)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--trackio", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--episodes", type=int, nargs=2, metavar=("START", "END"),
                   help="inclusive episode index range, e.g. 2000 2049 for metaworld push-v3")
    p.add_argument("--drop-features", nargs="*", default=None,
                   help="dataset input features to exclude, e.g. observation.environment_state")
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--n-action-steps", type=int, default=None)
    a = p.parse_args()
    episodes = list(range(a.episodes[0], a.episodes[1] + 1)) if a.episodes else None

    ckpt = start_training(
        repo_id=a.repo_id, root=a.root, output_dir=a.output_dir, policy_name=a.policy,
        job_name=a.job_name, device=a.device, steps=a.steps, batch_size=a.batch_size,
        log_freq=a.log_freq, save_freq=a.save_freq, env_eval_freq=a.env_eval_freq,
        num_workers=a.num_workers, trackio=a.trackio, overwrite_checkpoint=a.overwrite,
        episodes=episodes, horizon=a.horizon, n_action_steps=a.n_action_steps,
        drop_features=a.drop_features,
    )
    print(f"Checkpoint: {ckpt}")
