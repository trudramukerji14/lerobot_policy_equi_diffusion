"""Train a LeRobot policy (ported to lerobot 0.7.0).

Examples:
  # quick local smoke test
  uv run python train.py --steps 20 --batch-size 4 --device cpu --overwrite

  # real run on a rented GPU
  uv run python train.py --steps 200000 --batch-size 64 --device cuda --num-workers 8 --trackio
"""
import argparse
import shutil
import sys
from pathlib import Path

from lerobot.configs import parser

from lerobot.configs.default import DatasetConfig, EvalConfig, TrackioConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.envs.configs import MetaworldEnvConfig
from lerobot.optim.ema import WarmupEMAConfig
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
    ema=True,
    env_task=None,
    eval_episodes=10,
    resize_shape=None,
    crop_ratio=1.0,
    eval_batch_size=1,
    eval_max_episode_steps=None,
    resume_from=None,
    eval_async_envs=True,
):
    if resume_from:
        if overwrite_checkpoint:
            raise ValueError("--overwrite would delete the run being resumed; drop it when using --resume.")
        # LeRobot resolves the checkpoint to resume from by re-reading `--config_path` out of sys.argv
        # (see TrainPipelineConfig._resolve_resume_checkpoint), so make sure it is there.
        if parser.parse_arg("config_path") is None:
            sys.argv.append(f"--config_path={resume_from}")
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
    if resize_shape is not None:
        # The config derives crop_shape from resize_shape/crop_ratio only in __post_init__ (already run),
        # so set all three here. Random crop at train time, center crop at eval.
        policy.resize_shape = tuple(resize_shape)
        policy.crop_ratio = crop_ratio
        policy.crop_shape = (
            (int(resize_shape[0] * crop_ratio), int(resize_shape[1] * crop_ratio)) if crop_ratio < 1.0 else None
        )
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
        # Paper: EMA is used. WarmupEMAConfig defaults are the Diffusion Policy reference (power 0.75, max 0.9999).
        ema=WarmupEMAConfig() if ema else None,
        # In-training sim eval needs an env; only built when requested.
        env=MetaworldEnvConfig(task=env_task) if env_task else None,
        # batch_size > 1 steps that many eval episodes in parallel, so one batched policy call serves
        # all of them (a single episode is ~3 min on GPU, mostly the 100-step denoising per action chunk).
        eval=EvalConfig(
            n_episodes=eval_episodes,
            batch_size=eval_batch_size,
            max_episode_steps=eval_max_episode_steps,
            use_async_envs=eval_async_envs,
        ),
        resume=bool(resume_from),
    )
    if resume_from:
        # validate() skips the policy's optimizer/scheduler presets on resume, expecting them from the
        # checkpoint's train_config.json (which we don't load). Set them here; the optimizer state and
        # scheduler position themselves are restored from the checkpoint's training_state.
        train_config.optimizer = policy.get_optimizer_preset()
        train_config.scheduler = policy.get_scheduler_preset()

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
    p.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True,
                   help="EMA of weights (paper uses it); --no-ema to disable")
    p.add_argument("--env-task", default=None,
                   help="metaworld task for in-training sim eval, e.g. metaworld-push-v3")
    p.add_argument("--eval-episodes", type=int, default=10, help="episodes per in-training eval")
    p.add_argument("--resize-shape", type=int, nargs=2, metavar=("H", "W"), default=None,
                   help="resize images before the backbone (native metaworld is 480x480, too big for batch 128)")
    p.add_argument("--crop-ratio", type=float, default=1.0,
                   help="crop = resize * ratio (random in training); 1.0 disables cropping")
    p.add_argument("--eval-batch-size", type=int, default=1,
                   help="eval episodes run in parallel per in-training eval (use = --eval-episodes)")
    p.add_argument("--eval-max-episode-steps", type=int, default=None,
                   help="cap eval episode length (default: env default, 500 for metaworld)")
    p.add_argument("--eval-async-envs", action=argparse.BooleanOptionalAction, default=True,
                   help="run eval envs in worker processes; --no-eval-async-envs keeps them in-process "
                        "(needed on macOS, where OpenGL can't render in worker processes)")
    p.add_argument("--resume", default=None, metavar="CKPT",
                   help="resume from a checkpoint's pretrained_model dir or train_config.json; "
                        "use the same --job-name/--output-dir/--steps, and not --overwrite")
    a = p.parse_args()
    episodes = list(range(a.episodes[0], a.episodes[1] + 1)) if a.episodes else None

    ckpt = start_training(
        repo_id=a.repo_id, root=a.root, output_dir=a.output_dir, policy_name=a.policy,
        job_name=a.job_name, device=a.device, steps=a.steps, batch_size=a.batch_size,
        log_freq=a.log_freq, save_freq=a.save_freq, env_eval_freq=a.env_eval_freq,
        num_workers=a.num_workers, trackio=a.trackio, overwrite_checkpoint=a.overwrite,
        episodes=episodes, horizon=a.horizon, n_action_steps=a.n_action_steps,
        drop_features=a.drop_features, ema=a.ema, env_task=a.env_task,
        eval_episodes=a.eval_episodes, resize_shape=a.resize_shape, crop_ratio=a.crop_ratio,
        eval_batch_size=a.eval_batch_size, eval_max_episode_steps=a.eval_max_episode_steps,
        resume_from=a.resume, eval_async_envs=a.eval_async_envs,
    )
    print(f"Checkpoint: {ckpt}")
