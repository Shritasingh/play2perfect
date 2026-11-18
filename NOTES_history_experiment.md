# Notes: does adding history help the finetune ("perfect") stage?

Handoff notes from a Claude Code session (2026-11). Goal: keep the Stage-1 play policy
frozen as the initialization, and test whether adding history to Stage-2 precise-assembly
finetuning improves performance.

## Codebase map (short)

- `isaacsimenvs/train.py` — single training entry point (both stages). Boots Isaac Lab
  `AppLauncher`, merges Hydra config, `gym.make`, then the vendored `rl_games` Runner.
- `isaacsimenvs/tasks/play/` — Stage-1 `PlayEnv` (thin `DirectRLEnv`; logic lives in `utils/`:
  scene, reset, action, obs, reward, termination, goal sampling).
- `isaacsimenvs/tasks/precise_assembly/` — Stage-2 `PreciseAssemblyEnv(PlayEnv)`; goal is a
  hole-frame insertion-pose sequence; problem chosen by `env.precise_assembly.problem=<key>`.
- `isaacsimenvs/cfg/{task,train}/` — Hydra YAMLs. SAPG = PPO + flags in `rl_games/.../a2c_common.py`.
- `evaluation/problems/__init__.py` — registry of six released problems.
- `evaluation/eval_isaacsim.py` (viser GUI), `evaluation/eval_offline.py` (headless success rates).

## Is history already there? Yes, recurrent only

- Policy is an **LSTM** (1024 units, 1 layer, `before_mlp: True`, layer norm) followed by an
  MLP `[1024, 1024, 512, 512]`. See `isaacsimenvs/cfg/train/PreciseAssemblySAPG.yaml`
  (`network.rnn`). Identical to the Play config, which finetuning from the play checkpoint requires.
- **No explicit history**: no frame stacking or history buffer in `isaacsimenvs/`.
  `_stack_obs_dict` (`obs_utils.py`) concatenates fields at one timestep, not across time.
  The LSTM sees only the current observation each step.
- Training memory span: `seq_length: 16`, `horizon_length: 16` => truncated BPTT over 16 steps
  (~0.27 s at 60 Hz). Hidden state is carried across rollouts and zeroed on done
  (`zero_rnn_on_done`, `a2c_common.py`), so inference memory can exceed 16 steps.

### Actor observation (`obs.obs_list`, `cfg/task/PreciseAssembly.yaml`)

joint_pos, joint_vel (noisy), prev_action_targets (1-step action history), palm_pos, palm_rot,
object_rot (noisy), fingertip_pos_rel_palm, keypoints_rel_palm (noisy), keypoints_rel_goal (noisy),
object_scales.

Critic (`obs.state_list`, MLP, privileged) additionally has palm_vel, object_vel,
closest_keypoint_max_dist, closest_fingertip_dist, lifted_object, progress, successes, reward.

The actor has no object/palm velocity, no object position (only via keypoints), no contact/force.

### Temporal DR already in the env (`domain_randomization` in the same YAML)

obs delay up to 3 steps, action delay up to 3, object-state delay up to 10 (plus 1 cm position /
5 deg rotation noise), joint-vel noise 0.1. Delays are resampled per step (`_sample_delay` in
`obs_utils.py`), so observations are jittery; history could help filter that.

## Experiment plan

1. **Baseline**: released config (LSTM, 16-step BPTT).
2. **Longer BPTT only**: raise `seq_length` / `horizon_length` (e.g. 32, 64) in
   `PreciseAssemblySAPG.yaml`. Config-only; the play checkpoint still loads. Needs
   `horizon_length % seq_length == 0` and the minibatch assertion in `a2c_common.py` (~line 537);
   more memory per update, so `num_envs` / minibatch may need lowering. Caveat: also changes the
   PPO rollout length, so not a perfectly clean comparison.
3. **Explicit frame stacking** of the last k obs (k ~ 4), restricted to the noisy/delayed fields:
   keypoints_rel_goal, keypoints_rel_palm, prev_action_targets. Keep the current obs in its existing
   input slots, add the history as new columns with **zero-initialized** weights so step 0 behaves
   exactly like the play policy. Touch points: `compute_obs_dim` and the obs-space setup in
   `PlayEnv.__init__`, a history buffer in `build_observations` (flush on reset like `_sample_delay`),
   and checkpoint loading (extend LSTM input weights `weight_ih` and the running mean/std).
   **Not yet verified**: how the checkpoint is loaded in `train.py` / rl_games and how hard the
   padding is. Trace this first. SAPG also appends a block-id column to obs; keep that in mind.
4. **Both** 2 and 3, if time allows.

Use the same seed and finetuning budget; try one or two hard problems, e.g. `tight_insertion` and
`beam_assembly_step2`. Caveat to agree on up front: an LSTM can in principle learn what frame
stacking provides, so a null result is plausible.

## Running on a cluster

Needs the Isaac Sim / Isaac Lab env (`docs/isaacsim_installation.md`, Python 3.11) and a large GPU.
Default is 8192 envs; reduce on smaller GPUs. Checkpoints: `python download_checkpoints.py`.
Finetune example (README): `python isaacsimenvs/train.py --task Isaacsimenvs-PreciseAssembly-Direct-v0
--agent rl_games_cfg_entry_point --checkpoint pretrained_policy/model.pth
env.precise_assembly.problem=tight_insertion --headless` (SAPG: `--agent rl_games_sapg_cfg_entry_point`).
