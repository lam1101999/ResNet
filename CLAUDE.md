# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Reimplementation of **He et al., "Deep Residual Learning for Image Recognition", CVPR 2016** — covering both CIFAR-10 and ImageNet variants of ResNet, PlainNet, and VGG baselines.

## Commands

All commands use `uv run` since the project is managed with [uv](https://github.com/astral-sh/uv).

```bash
# Install / sync dependencies
uv sync

# Forward-pass shape check for every model
uv run python main.py --verify

# Train CIFAR-10 (default: resnet56, 164 epochs)
uv run python main.py --model resnet56 --dataset cifar10 --epochs 164

# Train ImageNet (requires HF login first)
uv run hf auth login
uv run python main.py --model resnet50 --dataset imagenet --epochs 90 --batch-size 256

# Resume an interrupted run
uv run python main.py --model resnet56 --dataset cifar10 --epochs 164 --resume

# Evaluate a saved checkpoint (no training)
uv run python main.py --model resnet56 --dataset cifar10 --eval-only --checkpoint weights/resnet56_best.pth
```

Key `main.py` flags: `--model`, `--dataset` (`cifar10`|`imagenet`), `--epochs`, `--batch-size`, `--lr`, `--milestones`, `--workers`, `--device` (`auto`|`cpu`|`cuda`|`mps`), `--weights-dir`, `--plot-dir`, `--resume`, `--eval-only`, `--checkpoint`.

## Architecture

### Model registry pattern

Each model file exposes factory functions **and** a `MODELS` dict for its supported variants:

| File | Dict exported | Models |
|------|--------------|--------|
| `src/model/resnet.py` | `IMAGENET_RESNET_MODELS` | resnet18/34/50/101/152 |
| `src/model/resnet_cifar.py` | `CIFAR_RESNET_MODELS` | resnet20/32/44/56/110 |
| `src/model/plain_net.py` | `IMAGENET_PLAIN_MODELS`, `CIFAR_PLAIN_MODELS` | plain18/34, plain20/32/44/56/110 |
| `src/model/vgg.py` | `IMAGENET_VGG_MODELS` | vgg16/16_bn/19/19_bn |

`src/model/__init__.py` merges these into two top-level dicts: **`CIFAR_MODELS`** and **`IMAGENET_MODELS`**. `main.py` imports only these two dicts — it never enumerates model names itself.

To add a new model: implement it in the appropriate file, add it to that file's `MODELS` dict, and it automatically appears everywhere.

### Training + checkpointing (`src/train.py`)

`train()` accepts `start_epoch`, `best_val_acc`, and `history` for seamless resume. It writes two files per run under `weights/`:
- `<model>_last.pth` — saved every epoch (full state: model + optimizer + scheduler + history)
- `<model>_best.pth` — saved only when val accuracy improves

`save_checkpoint` / `load_checkpoint` are module-level helpers importable independently.

### Data loaders (`src/data/`)

- `cifar.py` → `get_cifar10_loaders()` — uses torchvision, auto-downloads
- `imagenet.py` → `get_imagenet_loaders()` — wraps HuggingFace `ILSVRC/imagenet-1k` via a thin `HFImageNetDataset(Dataset)` adapter; requires prior `hf auth login` and accepting dataset terms

### Paper fidelity

- **CIFAR-10**: 3×3 first conv (no maxpool), 3 stages (channels 16→32→64), 6n+2 total layers
- **ImageNet**: 7×7 stem + maxpool, 4 stages, Option B projection shortcuts
- **PlainNet**: identical depth/width to ResNets but shortcuts removed (degradation baseline)
- Optimizer: SGD, momentum=0.9, weight_decay=1e-4, lr=0.1 divided by 10 at epochs 82 and 123 (MultiStepLR)

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **ResNet** (109 symbols, 202 relationships, 1 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/ResNet/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/ResNet/context` | Codebase overview, check index freshness |
| `gitnexus://repo/ResNet/clusters` | All functional areas |
| `gitnexus://repo/ResNet/processes` | All execution flows |
| `gitnexus://repo/ResNet/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
