# Auto-Formalizing Neuro-Symbolic Predictors

Code and data for **auto-nesy-bench**, a Neuro-Symbolic (NeSy) benchmark for turning natural-language domain knowledge into constraints written in conjunctive normal form formulas with LLMs, and for measuring what is their impact on NeSy predictors.

> See the benchmark [website](https://unitn-sml.github.io/auto-nesy-bench/) for more information!

The pipeline has two stages:

1. **Auto-formalization.** An LLM receives a textual description of a
   constraint and the list of its variables and writes the constraint in one
   of five formats: DIMACS, natural-language CNF (NAT), or a PySAT, CPMpy or
   SymPy program. The result is converted to DIMACS and compared with the ground
   truth by model counting (precision, recall, F1, accuracy).
2. **NeSy prediction.** The generated CNF is compiled into a sentential
   decision diagram (SDD) and used either as a **Semantic Probabilistic Layer**
   (SPL, hard constraint) or as a **Semantic Loss** (SL, soft constraint), and
   compared with the same predictor built on the expert-written formula.

The LLM is also evaluated on three benchmarks: auto-nesy-bench, [SATBench](https://huggingface.co/datasets/LLM4Code/SATBench) and [DCP-Bench-Open](https://github.com/DCP-Bench/DCP-Bench-Open).

## Repository layout

```
fonesy/                         Python package (python -m fonesy ...)
  datasets/                     benchmark loaders (dummy, auto-nesy-bench, dcp-bench, satbench)
  datasets/tasks/               data of the NeSy tasks (images, text, video, features)
  evaluation/                   format verifiers, auto-fixers, model counting
  experiments/                  sub-commands: evaluate, metrics, compile, spl, sl, optuna
  llms/                         vLLM backend and a dummy LLM
  networks/                     backbones (Conv4, ResNet-18, MLP, Faster-RCNN, SlowFast, BERT)
  prompts/                      prompt and self-correction templates
  data/auto-nesy-bench/         the benchmark: descriptions, oracles, ground-truth DIMACS
  data/dummy/                   seven toy formulas to test the pipeline
ground-truth-dimacs/            builds the ground-truth CNF of every benchmark
scripts/                        data generators (Kand-Logic, CLE4EVR, ROAD-R frames)
Makefile                        dataset download and preparation
```

## Installation

Python 3.12, with a CUDA 12.6 build of PyTorch:

```bash
pip install -r requirements.txt
pip install -r requirements.dev.txt   # black (optional)
```

## auto-nesy-bench

Every task has a detailed and a non-detailed constraint description (`fonesy/data/auto-nesy-bench/inputs_detailed/`, `inputs_non_detailed/`), a Python oracle declaring its variables (`targets/<task>.py`) and a ground-truth CNF (`targets/<task>.dimacs`).

| Task | Variables | Clauses | Data | NeSy backbone |
|------|----------:|--------:|------|---------------|
| `chx` | 9 | 27 | ChestX-ray14, four findings | ResNet-18 |
| `fashion` | 10 | 46 | Fashion-MNIST | Conv4 |
| `bdd-oia-2` | 11 | 17 | BDD-OIA (move forward / stop) | MLP |
| `cle4evr` | 12 | 28 | CLEVR-style scenes | Faster-RCNN |
| `mn-add-bin` | 13 | 512 | MNIST addition, binary | Conv4 |
| `cifar10` | 15 | 157 | CIFAR-10 | ResNet-18 |
| `mn-mul-bin` | 15 | 712 | MNIST multiplication, binary | Conv4 |
| `sushi` | 16 | 56 | SUSHI preferences | MLP |
| `kand-logic-2` | 24 | 1,328 | Kandinsky patterns, 2 objects | Faster-RCNN |
| `warcraft` | 24 | 289 | Warcraft shortest path, 4x4 | Conv4 |
| `bdd-oia` | 25 | 31 | BDD-OIA | MLP |
| `cebab` | 25 | 1,070 | CEBaB reviews | BERT |
| `kand-logic` | 36 | 129,648 | Kandinsky patterns, 3 objects | Faster-RCNN |
| `mn-add` | 39 | 364 | MNIST addition, one-hot | Conv4 |
| `road-r` | 41 | 243 | ROAD-R | SlowFast-50 |
| `sudoku` | 64 | 400 | 4x4 visual Sudoku | Conv4 |
| `cifar100` | 100 | 4,951 | CIFAR-100 | ResNet-18 |
| `mn-mul` | 102 | 3,514 | MNIST multiplication, one-hot | Conv4 |

The ground-truth CNFs are compiled with PySAT by `ground-truth-dimacs/auto-nesy-bench.py` (the `road-r` CNF is the set of requirements released with ROAD-R) and checked against the oracles by `ground-truth-dimacs/check.py`, exhaustively for tasks with at most 20 variables:

```bash
make auto-nesy-bench
```

## Data

Nothing is downloaded automatically, except the torchvision datasets on first use. `make help` lists the targets; data is written to `fonesy/data/`.

| Target | Contents |
|--------|----------|
| `make satbench` | SATBench at a pinned revision, keeping the 1,041 satisfiable instances |
| `make dcp-bench` | DCP-Bench-Open, converted to CNF with CPMpy (`ground-truth-dimacs/dcp-bench.py`) |
| `make negations` | Optional cache of the negated auto-nesy-bench targets, which speeds up model counting |
| `make torchvision` | MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100 |
| `make sushi`, `make cebab`, `make warcraft` | SUSHI3, CEBaB v1.1, Warcraft II terrain tiles |
| `make road-r` | ROAD-R videos and annotations (needs `gdown`), then frame extraction |
| `make chx GCS_PROJECT=<project>` | ChestX-ray14 images and four-findings labels (needs an authenticated `gcloud`) |
| `make bdd-oia BDD_OIA_ZIP=<file or URL>` | BDD-OIA Faster-RCNN features released with rsbench (`bdd_2048.zip`) |
| `make kand-logic` | Kand-Logic pairs rendered with the rsbench generator (clones rsbench-code) |
| `make cle4evr` | CLE4EVR scenes rendered with Blender (needs `blender` and `xvfb`) |

SUSHI3 and the CIFAR datasets cannot be redistributed and are always fetched from their original sources.

## Usage

Every command has the form `python -m fonesy OUTPUT_DIR [--seed N] <command> ...` and prints its options with `--help`. Results (`*.results.json`, `*.stats.json`, `*.metrics.csv`) and logs are written to `OUTPUT_DIR`.

### 1. Auto-formalization

```bash
python -m fonesy results --seed 32 evaluate --cnf-format cpmpy --budget 5 \
    auto-nesy-bench --mode zero-shot-cot \
    vllm --model-path gpt-oss-120b --tensor-parallel-size 1
```

- Benchmark: `auto-nesy-bench` (add `--non-detailed` for the non-detailed  descriptions), `satbench`, `dcp-bench`, or `dummy` (toy formulas).
- `--cnf-format`: `dimacs`, `nat`, `pysat`, `cpmpy` or `sympy`. 
- `--mode`: `zero-shot` (ZS) or `zero-shot-cot` (ZS-CoT).
- `--budget`: number of self-verification rounds (5 in the paper).
- LLM: `vllm --model-path <shortcut or Hugging Face id>` or `dummy` (fixed answer, no GPU). The evaluated models have shortcuts: `qwen3-8b`, `qwen3-32b`, `qwen3-coder-next`, `mistral-nemo`, `gemma3-27b`, `gemma4-31b`, `phi4-reasoning-plus`, `olmo3-32b-think`, `deepseek-r1-llama-70b`, `gpt-oss-20b`, `gpt-oss-120b`. Generation is capped at 32,768 new tokens.

Model counting (exact up to 20 variables, ApproxMC with ε=0.1, δ=0.05 above) runs in background processes while the LLM generates (`--mc-workers`, default
4). Every sample is logged to `<experiment>.samples.jsonl` and every valid formula is saved to `OUTPUT_DIR/cnf/<experiment>/<task>.cnf`, together with a `.meta.json` file giving its number of auxiliary variables.

To recompute the metrics of a finished run without querying the LLM again, run `metrics` with the same arguments (`--epsilon`, `--delta` and `--model-count-timeout` configure the counter):

```bash
python -m fonesy results --seed 32 metrics --cnf-format cpmpy --budget 5 \
    auto-nesy-bench --mode zero-shot-cot \
    vllm --model-path gpt-oss-120b --tensor-parallel-size 1
```

### 2. Downstream NeSy predictors

`spl` trains a backbone and an SPL head jointly; `sl` trains a backbone and an MLP classifier with the semantic loss (`--sl-weight 0` gives the unconstrained baseline). The constraint is given with `--from-dimacs`: either the ground truth or a formula saved by `evaluate`.

```bash
# Ground-truth formula
python -m fonesy results --seed 1011 spl --spl-task sudoku \
    --from-dimacs fonesy/data/auto-nesy-bench/targets/sudoku.dimacs \
    --spl-epochs 50 --spl-lr 2.9e-4 --spl-batch-size 16 --spl-hidden 455 \
    auto-nesy-bench dummy

# Formula generated in step 1
python -m fonesy results --seed 1011 spl --spl-task sudoku \
    --from-dimacs results/cnf/<experiment>/sudoku.cnf --auxiliary-num-vars <n> \
    --spl-epochs 50 --spl-lr 4.0e-4 --spl-batch-size 16 --spl-hidden 227 \
    auto-nesy-bench dummy

# Semantic loss
python -m fonesy results --seed 1011 sl --spl-task sudoku \
    --from-dimacs fonesy/data/auto-nesy-bench/targets/sudoku.dimacs \
    --spl-epochs 50 --spl-lr 2.9e-4 --spl-batch-size 16 --spl-hidden 455 \
    --sl-weight 1.0 auto-nesy-bench dummy
```

Both report accuracy, macro precision/recall/F1, AUROC/AUPRC and the **consistency** of the predictions with the ground-truth formula. With `--from-dimacs`, `spl` also reports the relation (REL) between the formula and the ground truth (`equal`, `permissive`, `too_strict` or `incomparable`) and the ratio of their model counts (MC-R). 

Hyperparameters (learning rate, head width, batch size) are selected with Optuna (TPE sampler, median pruner, best validation macro-F1):

```bash
python -m fonesy results --seed 32 optuna --spl-task sudoku \
    --from-dimacs fonesy/data/auto-nesy-bench/targets/sudoku.dimacs \
    --n-trials 30 --spl-epochs 50 --batch-sizes 16,32,64,128,256 \
    auto-nesy-bench dummy
```

The paper retrains the selected configuration for 50 epochs (10 for `road-r`) with seeds 1011, 2021, 3031, 4041 and 5051; the selected hyperparameters are listed in its appendix.

## Acknowledgments

`auto-nesy-bench` builds on [rsbench](https://unitn-sml.github.io/rsbench) and on the following datasets and benchmarks. Please also cite them when using te hcorresponding tasks:

- [MNIST](http://yann.lecun.com/exdb/mnist/) and MNIST addition
- ([DeepProbLog](https://github.com/ML-KULeuven/deepproblog)),
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist),
- [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html),
- [BDD-OIA](https://twizwei.github.io/bddoia_project/),
- [ROAD-R](https://github.com/EGiunchiglia/ROAD-R),
- [SUSHI](https://www.kamishima.net/sushi/),
- [CEBaB](https://cebabing.github.io/CEBaB/),
- [ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) with the
- [four-findings labels](https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest),
- [Kandinsky patterns](https://github.com/human-centered-ai-lab/dat-kandinsky-patterns),
- [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/), visual Sudoku (Augustine et al., 2022) and Warcraft shortest paths
- ([Vlastelica et al., 2020](https://github.com/martius-lab/blackbox-backprop)).
- [SATBench](https://github.com/Anjiang-Wei/SATBench) and
- [DCP-Bench-Open](https://github.com/DCP-Bench/DCP-Bench-Open) are the other two auto-formalization benchmarks.

The pipeline relies on:

- [vLLM](https://github.com/vllm-project/vllm),
- [PySAT](https://github.com/pysathq/pysat), 
- [CPMpy](https://github.com/CPMpy/cpmpy),
- [SymPy](https://www.sympy.org), 
- [ApproxMC](https://github.com/meelgroup/approxmc),
- [PySDD](https://github.com/wannesm/PySDD), 
- [cirkit](https://github.com/april-tools/cirkit),
- [KLay](https://github.com/ML-KULeuven/klay)
- [Optuna](https://optuna.org).

## Contact

For questions about the code or the paper, open an issue or contact:

- [Samuele Bortolotti](mailto:samuele.bortolotti@unitn.it)
- [Weixin Chen](mailto:weixinc2@illinois.edu)

## License

The code is released under the BSD-3-Clause license (see `LICENSE`). The redistributed data follow the licenses of their sources; ROAD-R, the most restrictive one, is CC BY-NC-SA 4.0.

## Citation

```bibtex
@misc{bortolotti2026autoformalizing,
  title  = {Auto-Formalizing Neuro-Symbolic Predictors},
  author = {Bortolotti, Samuele and Chen, Weixin and Zhao, Han and
            Passerini, Andrea and Teso, Stefano and Vergari, Antonio},
  year   = {2026},
  note   = {Preprint}
}
```
