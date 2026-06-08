# Running the pending detectors

Six wrapped detectors are marked `pending` in `leaderboard.csv` because their full
matched-27k evaluation exceeds the 5-hour budget or needs a backbone/env that is not yet
staged. This is the recipe to run each when you want its row.

All of these are **new** models (not in the paper), so vidaudit-computed rows are fine to
add. Do NOT re-run the published detectors (ReStraV/D3/WaveRep/NSG-VD/FVMD/RAFT/CLIP/
TemporalSpec) through this engine: the repo leaderboard mirrors the paper's authoritative
numbers, and re-running them only diverges them by split-path noise.

## Shared setup (cluster)

- Matched-27k cell: `/N/project/de_briujn_graph/Projects/mv_vid_classify/Paper/Results/baseline_clip_subset.csv` (27,000 clips, `mp4_path` resolved).
- Caches off HOME: `export HF_HOME=/N/scratch/meocakir/hf_cache HUGGINGFACE_HUB_CACHE=$HF_HOME/hub MODELSCOPE_CACHE=/N/scratch/meocakir/ms_cache TORCH_HOME=/N/scratch/meocakir/torch_cache`.
- The `gpu` partition has a large backlog: submit each model as its OWN sbatch so they queue in parallel. `extract` is resumable (checkpoints every 200 clips, skips done ids), so a job that hits its walltime just re-submits and continues.
- Score path: `extract --kind score` then `eval --scores <csv>`; features path: `extract --kind features` then `eval --features <csv>`. Fold the resulting `logo_ood`/`rvr`/operating points into `leaderboard.csv` and `run.py leaderboard`.

## MLLMs: Skyra, VideoVeritas, Ivy-xDetector (the cost problem)

At HuggingFace `generate()` speed these are ~50-90 s/clip (7-8B) = ~190-500 GPU-hr each on
the full cell. The fix is **batched inference with vLLM** (~10-30x throughput -> ~15-30
GPU-hr/model).

1. Build a dedicated env (do NOT reuse the py3.14 / transformers-5.10 vidaudit env): py3.11
   or 3.12 + a vLLM build whose pinned transformers supports Qwen2.5-VL / Qwen3-VL. Use the
   cluster env-build pattern (`CONDA_PKGS_DIRS` + `-p` prefix off HOME; project storage).
2. Run vLLM offline batched generation over the 27k cell with each model's exact prompt and
   verdict tag, then map the verdict logits to p(generated) (fake vs real) per clip:
   - Skyra: `JoeLeelyf/Skyra-RL` (HF), 16 uniform frames, `<answer>Fake/Real</answer>`.
   - VideoVeritas: `EricTanh/VideoVeritas` (ModelScope), 3 fps, `<answer>real/fake</answer>`.
   - Ivy-xDetector: `AI-Safeguard/Ivy-Fake` (HF, 3B), 1 fps, `<conclusion>real/fake</conclusion>`.

   Frame sampling note: the current HF wrappers (`mllm.py`) sample a uniform `n_frames=16`
   for all three, so only Skyra matches its native protocol. For a faithful vLLM run, match
   each model's native rate (VideoVeritas 3 fps, Ivy 1 fps at max 6 frames) or keep the
   uniform-16 approximation and footnote it on the leaderboard row.
3. **Verify the tie first.** These checkpoints omit a tied `lm_head`; transformers-5.x left
   it random (garbage output) and we copy the input embeddings into it (see
   `vidaudit/detectors/mllm.py._load`). Confirm vLLM loads `lm_head` correctly on a few
   clips; if the output is garbage, apply the same embedding->lm_head copy in the vLLM path.
4. Write a per-clip `score` CSV, then `eval --scores`. The adapter's soft-score logic
   (`score()` reads the last verdict token's fake-vs-real logits) is validated on A100 for
   the HF path and can be ported, or just take vLLM's logprobs over the verdict tokens.

If a smaller pass is acceptable, run on a balanced ~2k subset of the cell first (footnote it
as a subsample).

## AIGVDet (two-stream, ~7 h)

Weights (academic Google Drive): `gdown 10EXwX9cXR0VuBmWq7QpMfotnPtIRKIsV -O checkpoints/original.pth`
and `gdown 1MiMkASZ-SDisCuLi-A7R-Yvqjzsy_BMC -O checkpoints/optical.pth`. The decode cap
(`_extract._MAX_DECODE`) already fixes the OOM that killed the first run at 3k/27k. Options:

- One 8-hour `gpu` job, resumable: `python run.py extract aigvdet --manifest <cell> --out aigvdet_score.csv --kind score --weights checkpoints` then `eval --scores`. It is ~7 h, over the 5 h budget but a single job tolerates it.
- For wall-clock, split the cell into N chunks and submit an array job (one fresh process per chunk, each writing its own CSV; concatenate, then `eval`).
- The optical branch uses torchvision RAFT (approximates the paper's raft-things); spatial-only (`--weights <single original.pth>`) is much faster but not the published two-stream method.

## L3DE (non-commercial, heavy)

Needs a separate env with UniDepth-v2 (CC-BY-NC-4.0; `pip install unidepth` + its xformers /
CUDA op), plus DINOv2-ViT-G (torch.hub, auto) and the L3DE checkpoint
(`gdown 1wBAAsJPcsT_bIKXetDbd23PKjmCUtb5s -O weights/L3DE.pth`). Flag the row non-commercial.
Per-clip cost is high (three backbones); expect well over 5 h on the full cell.

## STALL (gated backbone)

The VATEX calibration is in the repo
(`wget https://raw.githubusercontent.com/OmerBenHayun/STALL/main/precomputed/stall_params_vatex_dino_v3.npz`),
but DINOv3 ViT-L/16 is gated: request access at Meta's DINOv3 downloads page (or HF
`facebook/dinov3-vitl16-pretrain-lvd1689m`), place the `.pth`, and pass
`STALL(dinov3_dir=<clone of facebookresearch/dinov3>, dinov3_weights=<.pth>)`. Load the
released calibration npz via `load_weights`, then `extract --kind score` (the native
`1 - mean-percentile` readout, which is the `native-head` row the leaderboard lists). Its
`features()` instead emits the 4-d evidence vector [spatial %, temporal %, spatial LL,
temporal LL] for the L2-LR readout if you want that variant.
