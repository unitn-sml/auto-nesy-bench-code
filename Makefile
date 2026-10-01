# Download and prepare the data used by the experiments.
#
#   make help        list the targets
#   make benchmarks  LLM auto-formalization benchmarks (SATBench, DCP-Bench)
#   make nesy        data of the NeSy (SPL / SL) tasks that can be fetched automatically
#
# auto-nesy-bench itself (descriptions, oracles and ground-truth DIMACS) is
# part of the repository: `make auto-nesy-bench` only re-derives its DIMACS.

PYTHON       ?= python
DATA         := fonesy/data
RSBENCH_PATH ?= external/rsbench-code

.PHONY: help benchmarks nesy auto-nesy-bench negations satbench dcp-bench \
        torchvision sushi cebab warcraft chx road-r bdd-oia kand-logic cle4evr rsbench-code

help:
	@echo "LLM benchmarks"
	@echo "  auto-nesy-bench  recompile the ground-truth DIMACS and check them against the oracles"
	@echo "  negations        cache the negated auto-nesy-bench targets (speeds up model counting)"
	@echo "  satbench         download SATBench and keep the 1,041 satisfiable instances"
	@echo "  dcp-bench        download DCP-Bench-Open and convert it to CNF"
	@echo "NeSy tasks (SPL / SL)"
	@echo "  torchvision      MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100 (also fetched on first use)"
	@echo "  sushi cebab warcraft road-r kand-logic"
	@echo "  chx              needs gcloud and GCS_PROJECT=<billing project>"
	@echo "  bdd-oia          needs BDD_OIA_ZIP=<path or URL of rsbench's bdd_2048.zip>"
	@echo "  cle4evr          needs Blender and xvfb"

benchmarks: satbench dcp-bench negations

nesy: torchvision sushi cebab warcraft road-r kand-logic

# ── auto-nesy-bench ─────────────────────────────────────────────────────────

auto-nesy-bench:
	$(PYTHON) ground-truth-dimacs/auto-nesy-bench.py $(DATA)/auto-nesy-bench/targets
	$(PYTHON) ground-truth-dimacs/check.py

negations:
	$(PYTHON) ground-truth-dimacs/negate.py

# ── SATBench (Wei et al., 2025), pinned Hugging Face revision ───────────────

satbench:
	$(PYTHON) ground-truth-dimacs/satbench.py all

# ── DCP-Bench-Open (Michailidis et al.) ─────────────────────────────────────

DCPBENCH_VERSION ?= v0.1.1

dcp-bench:
	mkdir -p $(DATA)/dcp-bench/raw
	curl -L -o $(DATA)/dcp-bench/raw/dcp-bench-open.jsonl \
	    https://github.com/DCP-Bench/DCP-Bench-Open/releases/download/$(DCPBENCH_VERSION)/dcp-bench-open.jsonl
	$(PYTHON) ground-truth-dimacs/dcp-bench.py

# ── NeSy tasks ──────────────────────────────────────────────────────────────

torchvision:
	$(PYTHON) -c "from torchvision import datasets as d; \
	[c('$(DATA)', train=t, download=True) for c in (d.MNIST, d.FashionMNIST, d.CIFAR10, d.CIFAR100) for t in (True, False)]"

# SUSHI3 (Kamishima, 2003): research use only, not redistributable.
sushi:
	mkdir -p $(DATA)/sushi/raw
	curl -sL -o $(DATA)/sushi/raw/sushi3-2016.zip https://www.kamishima.net/asset/sushi3-2016.zip
	cd $(DATA)/sushi/raw && unzip -o -q sushi3-2016.zip

# CEBaB v1.1 (Abraham et al., 2022).
cebab:
	mkdir -p $(DATA)/cebab
	curl -sL -o $(DATA)/cebab/CEBaB-v1.1.zip https://raw.githubusercontent.com/CEBaBing/CEBaB/gh-pages/CEBaB-v1.1.zip
	cd $(DATA)/cebab && unzip -o -q -j CEBaB-v1.1.zip \
	    "CEBaB-v1.1/train_observational.json" "CEBaB-v1.1/dev.json" "CEBaB-v1.1/test.json"

# Warcraft II terrain tiles, from the SP(12) Warcraft maps.
warcraft:
	mkdir -p $(DATA)/warcraft
	curl -sL -o $(DATA)/warcraft/train.npz \
	    https://github.com/archettialberto/tilebased_navigation_datasets/raw/main/warcraft/train.npz

# ChestX-ray14 images with the four-findings labels (Majkowska et al., 2020).
# The labels are in a Requester-Pays bucket: requires an authenticated
# `gcloud` and a billing project. SKIP_IMAGES=1 skips the ~42 GB of images.
CHX_BUCKET := gs://gcs-public-data--healthcare-nih-chest-xray/four_findings_expert_labels
CHX_CSVS   := individual_readers test_labels validation_labels
CHX_IMAGES := vfk49d74nhbxq3nqjg0900w5nvkorp5c i28rlmbvmfjbl8p2n3ril0pptcmcu9d1 \
              f1t00wrtdk94satdfb9olcolqx20z2jp 0aowwzs5lhjrceb3qp67ahp0rd1l1etg \
              v5e3goj22zr6h8tzualxfsqlqaygfbsn asi7ikud9jwnkrnkj99jnpfkjdes7l6l \
              jn1b4mw4n6lnh74ovmcjb8y48h8xj07n tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j \
              upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj l6nilvfa9cg3s28tqv1qc1olm3gnz54p \
              hhq8fkdgvcari67vfhs7ppg2w6ni4jze ioqwiy20ihqwyr8pf4c24eazhh281pbu

chx:
	@test -n "$(GCS_PROJECT)" || { echo "Set GCS_PROJECT=<billing project>"; exit 1; }
	mkdir -p $(DATA)/chx/images
	for csv in $(CHX_CSVS); do \
	    gcloud storage cp -U --billing-project $(GCS_PROJECT) \
	        $(CHX_BUCKET)/four_findings_expert_labels_$$csv.csv $(DATA)/chx/; \
	done
	if [ "$(SKIP_IMAGES)" != "1" ]; then \
	    for id in $(CHX_IMAGES); do \
	        curl -sL https://nihcc.box.com/shared/static/$$id.gz \
	            | tar -xz -C $(DATA)/chx/images --strip-components=1; \
	    done; \
	fi

# ROAD-R (Giunchiglia et al., 2023): train-val videos + annotations (~4.6 GB),
# then the 8-frame clips of every 10th annotated frame (~13 GB).
ROAD_R_FOLDER := https://drive.google.com/drive/folders/1hCLlgRqsJBONHgwGPvVu8VWXxlyYKCq-

road-r:
	mkdir -p $(DATA)/road-r/videos
	cd $(DATA)/road-r && gdown --folder --continue $(ROAD_R_FOLDER)
	mv -n $(DATA)/road-r/public-train-val-set/videos/*.mp4 $(DATA)/road-r/videos/
	mv -n $(DATA)/road-r/public-train-val-set/road_trainval_v1.0.json $(DATA)/road-r/
	rm -rf $(DATA)/road-r/public-train-val-set
	$(PYTHON) scripts/extract_road_r_frames.py --root $(DATA)/road-r --stride 10

# BDD-OIA 2048-d Faster-RCNN features released with rsbench (bdd_2048.zip).
bdd-oia:
	@test -n "$(BDD_OIA_ZIP)" || { echo "Set BDD_OIA_ZIP=<path or URL of bdd_2048.zip>"; exit 1; }
	mkdir -p $(DATA)/bdd-oia
	if [ -f "$(BDD_OIA_ZIP)" ]; then cp "$(BDD_OIA_ZIP)" $(DATA)/bdd-oia/bdd_2048.zip; \
	else curl -L -o $(DATA)/bdd-oia/bdd_2048.zip "$(BDD_OIA_ZIP)"; fi
	cd $(DATA)/bdd-oia && unzip -o -q bdd_2048.zip

rsbench-code:
	test -d $(RSBENCH_PATH) || git clone --depth 1 https://github.com/unitn-sml/rsbench-code.git $(RSBENCH_PATH)

# Kand-Logic pairs rendered with rsbench's Kandinsky generator (800/100/100).
kand-logic: rsbench-code
	$(PYTHON) scripts/generate_kand_logic.py --rsbench-path $(RSBENCH_PATH) \
	    --variant kand-logic --output-dir $(DATA)/kand-logic \
	    --n-train 800 --n-val 100 --n-test 100 --pool-size 4000 --seed 0
	$(PYTHON) scripts/generate_kand_logic.py --rsbench-path $(RSBENCH_PATH) \
	    --variant kand-logic-2 --output-dir $(DATA)/kand-logic-2 \
	    --n-train 800 --n-val 100 --n-test 100 --pool-size 4000 --seed 0

# CLE4EVR scenes rendered with Blender/Cycles (5000/2000/5000).
cle4evr: rsbench-code
	xvfb-run -a blender -noaudio -b -P scripts/generate_cle4evr.py -- \
	    --output-dir $(DATA)/cle4evr \
	    --properties_json scripts/cle4evr_config/properties.json \
	    --n-train 5000 --n-val 2000 --n-test 5000 \
	    --render_num_samples 128 --use_gpu 1 --width 320 --height 240 --seed 0
