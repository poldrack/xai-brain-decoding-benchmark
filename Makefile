all: hyperopt train glm_bold interpret glm_attributions similarity faithfulness sanity_checks

hyperopt: results/hyperopt
train: results/models 
glm_bold: results/glm/BOLD
meta: results/meta_analysis
interpret: results/attributions
glm_attributions: results/glm/attributions
similarity: results/brain_map_similarity 
faithfulness: results/faithfulness 
sanity_checks: results/sanity_checks 

setup-env:
	micromamba create -y -f env.yml
	micromamba run -n xai pip install -r requirements.txt

train-heat:
	# heat-rejection
	python3 scripts/train.py \
		--task heat-rejection \
		--data-dir data/task-heat-rejection \
		--num-runs 10 \
		--num-folds 1 \
		--model-config results/hyperopt/task-heat-rejection/best_model_config.json \
		--run-group-name task-heat-rejection \
		--log-dir results/models \
		--report-to tensorboard

train-wm:
	# WM
	python3 scripts/train.py \
		--task WM \
		--data-dir data/task-WM \
		--num-runs 10 \
		--num-folds 1 \
		--model-config results/hyperopt/task-WM/best_model_config.json \
		--run-group-name task-WM \
		--log-dir results/models

train-motor:
	# MOTOR
	python3 scripts/train.py \
		--task MOTOR \
		--data-dir data/task-MOTOR \
		--num-runs 10 \
		--num-folds 1 \
		--model-config results/hyperopt/task-MOTOR/best_model_config.json \
		--run-group-name task-MOTOR_final-model-fits \
		--log-dir results/models


RESULTS_DIR = /Users/poldrack/data_unsynced/ray_storage
# HYPEROPT 3D-CNN model configurations
results/hyperopt: scripts/hyperopt.py scripts/identify-best-model-config.py
	# heat-rejection
	python3 scripts/hyperopt.py \
		--task heat-rejection \
		--data-dir data/task-heat-rejection \
		--log-dir $(RESULTS_DIR)/hyperopt/task-heat-rejection
	python3 scripts/identify-best-model-configuration.py \
		--task heat-rejection \
		--hyperopt-dir $(RESULTS_DIR)/hyperopt/task-heat-rejection

	# WM
	python3 scripts/hyperopt.py \
		--task WM \
		--data-dir data/task-WM \
		--log-dir results/hyperopt/task-WM
	python3 scripts/identify-best-model-configuration.py \
		--task WM \
		--hyperopt-dir results/hyperopt/task-WM

	# MOTOR
	python3 scripts/hyperopt.py \
		--task MOTOR \
		--data-dir data/task-MOTOR \
		--log-dir results/hyperopt/task-MOTOR
	python3 scripts/identify-best-model-configuration.py \
		--task MOTOR \
		--hyperopt-dir results/hyperopt/task-MOTOR

# TRAIN best-performing model configurations
results/models: scripts/train.py
	# heat-rejection
	python3 scripts/train.py \
		--task heat-rejection \
		--data-dir /Users/poldrack/Dropbox/code/xai-brain-decoding-benchmark/data/task-heat-rejection \
		--num-runs 10 \
		--num-folds 1 \
		--model-config results/hyperopt/task-heat-rejection/best_model_config.json \
		--run-group-name task-heat-rejection \
		--log-dir results/models

	# WM
	python3 scripts/train.py \
		--task WM \
		--data-dir data/task-WM \
		--num-runs 10 \
		--num-folds 1 \
		--model-config results/hyperopt/task-WM/best_model_config.json \
		--run-group-name task-WM \
		--log-dir results/models

	# MOTOR
	python3 scripts/train.py \
		--task MOTOR \
		--data-dir data/task-MOTOR \
		--num-runs 10 \
		--num-folds 1 \
		--model-config results/hyperopt/task-MOTOR/best_model_config.json \
		--run-group-name task-MOTOR_final-model-fits \
		--log-dir results/models

	# Figure
	python3 scripts/fig_decoding-performance.py

# GLM for BOLD data
run_glm_bold: # results/glm/BOLD: scripts/glm-BOLD_subject-level.py scripts/glm-BOLD_group-level.py
	# heat-rejection
	# python3 scripts/glm-BOLD_subject-level.py \
	# 	--task heat-rejection \
	# 	--data-dir data/task-heat-rejection \
	# 	--subject-level-maps-dir results/glm/BOLD/task-heat-rejection/subject_level
	# python3 scripts/glm-BOLD_group-level.py \
	# 	--task heat-rejection \
	# 	--subject-level-maps-dir results/glm/BOLD/task-heat-rejection/subject_level \
	# 	--group-level-maps-dir results/glm/BOLD/task-heat-rejection/group_level

	# WM
	python3 scripts/glm-BOLD_subject-level.py \
		--task WM \
		--data-dir data/task-WM \
		--subject-level-maps-dir results/glm/BOLD/task-WM/subject_level
	python3 scripts/glm-BOLD_group-level.py \
		--task WM \
		--subject-level-maps-dir results/glm/BOLD/task-WM/subject_level \
		--group-level-maps-dir results/glm/BOLD/task-WM/group_level

	# MOTOR
	python3 scripts/glm-BOLD_subject-level.py \
		--task MOTOR \
		--data-dir data/task-MOTOR \
		--subject-level-maps-dir results/glm/BOLD/task-MOTOR/subject_level
	python3 scripts/glm-BOLD_group-level.py \
		--task MOTOR \
		--subject-level-maps-dir results/glm/BOLD/task-MOTOR/subject_level \
		--group-level-maps-dir results/glm/BOLD/task-MOTOR/group_level

# ATTRIBUTE model decoding decisions for test data
results/attributions: scripts/interpret.py
	# heat-rejection
	python3 scripts/interpret.py \
		--task heat-rejection \
		--fitted-model-dir results/models/task-heat-rejection/best_model_dir \
		--data-dir data/task-heat-rejection \
		--attributions-dir results/attributions/task-heat-rejection

	# WM
	python3 scripts/interpret.py \
		--task WM \
		--fitted-model-dir results/models/task-WM/best_model_dir \
		--data-dir data/task-WM \
		--attributions-dir results/attributions/task-WM

	# MOTOR
	python3 scripts/interpret.py \
		--task MOTOR \
		--fitted-model-dir results/models/task-MOTOR/best_model_dir \
		--data-dir data/task-MOTOR \
		--attributions-dir results/attributions/task-MOTOR

# GLM for attribution data
results/glm/attributions: scripts/glm-attributions_subject-level.py scripts/glm-attributions_group-level.py
	# heat-rejection
	python3 scripts/glm-attributions_subject-level.py \
		--task heat-rejection \
		--attributions-dir results/attributions/task-heat-rejection \
		--subject-level-maps-dir results/glm/attributions/task-heat-rejection/subject
	python3 scripts/glm-attributions_group-level.py \
		--task heat-rejection \
		--subject-level-maps-dir results/glm/attributions/task-heat-rejection/subject \
		--group-level-maps-dir results/glm/attributions/task-heat-rejection/group

	# WM
	python3 scripts/glm-attributions_subject-level.py \
		--task WM \
		--attributions-dir results/attributions/task-WM \
		--subject-level-maps-dir results/glm/attributions/task-WM/subject
	python3 scripts/glm-attributions_group-level.py \
		--task WM \
		--subject-level-maps-dir results/glm/attributions/task-WM/subject \
		--group-level-maps-dir results/glm/attributions/task-WM/group

	# MOTOR
	python3 scripts/glm-attributions_subject-level.py \
		--task MOTOR \
		--attributions-dir results/attributions/task-MOTOR \
		--subject-level-maps-dir results/glm/attributions/task-MOTOR/subject
	python3 scripts/glm-attributions_group-level.py \
		--task MOTOR \
		--subject-level-maps-dir results/glm/attributions/task-MOTOR/subject \
		--group-level-maps-dir results/glm/attributions/task-MOTOR/group

# META-ANALYSIS with NeuroQuery
results/meta_analysis: scripts/meta-analysis.py
	python3 scripts/meta-analysis.py \
		--meta-maps-dir results/meta_analysis

# SIMILARITY of BOLD GLM maps and attribution GLM maps
results/brain_map_similarity: scripts/brain-map-similarities.py scripts/fig_brain-map-similarities.py
	# heat-rejection
	python3 scripts/brain_map_similarities.py \
		--task heat-rejection \
		--bold-glm-maps-dir results/glm/BOLD/task-heat-rejection \
		--attribution-glm-maps-dir results/glm/attributions/task-heat-rejection \
		--attributions-dir results/attributions/task-heat-rejection \
		--meta-maps-dir results/meta_analysis/task-heat-rejection \
		--brain-maps-similarity-dir results/brain_map_similarity/task-heat-rejection

	# WM
	python3 scripts/brain_map_similarities.py \
		--task WM \
		--bold-glm-maps-dir results/glm/BOLD/task-WM \
		--attribution-glm-maps-dir results/glm/attributions/task-WM \
		--attributions-dir results/attributions/task-WM \
		--meta-maps-dir results/meta_analysis/task-WM \
		--brain-maps-similarity-dir results/brain_map_similarity/task-WM

	# MOTOR
	python3 scripts/brain_map_similarities.py \
		--task MOTOR \
		--bold-glm-maps-dir results/glm/BOLD/task-MOTOR \
		--attribution-glm-maps-dir results/glm/attributions/task-MOTOR \
		--attributions-dir results/attributions/task-MOTOR \
		--meta-maps-dir results/meta_analysis/task-MOTOR \
		--brain-maps-similarity-dir results/brain_map_similarity/task-MOTOR

	# Figure
	python3 scripts/fig_brain-map-similarities.py \
		--mfx-dir results/mfx \
		--brain-maps-similarity-base-dir results/brain_map_similarity

# FAITHFULNESS of attributions
results/faithfulness: scripts/faithfulness.py scripts/fig_faithfulness.py
	# heat-rejection
	python3 scripts/faithfulness.py \
		--task heat-rejection \
		--fitted-model-dir results/models/task-heat-rejection \
		--data-dir data/task-heat-rejection \
		--attributions-dir results/attributions/task-heat-rejection \
		--faithfulness-dir results/faithfulness/task-heat-rejection

	# WM
	python3 scripts/faithfulness.py \
		--task WM \
		--fitted-model-dir results/models/task-WM \
		--data-dir data/task-WM \
		--attributions-dir results/attributions/task-WM \
		--faithfulness-dir results/faithfulness/task-WM

	# MOTOR
	python3 scripts/faithfulness.py \
		--task MOTOR \
		--fitted-model-dir results/models/task-MOTOR \
		--data-dir data/task-MOTOR \
		--attributions-dir results/attributions/task-MOTOR \
		--faithfulness-dir results/faithfulness/task-MOTOR

	# Figure
	python3 scripts/fig_faithfulness.py \
		--mfx-dir results/mfx \
		--faithfulness-base-dir results/faithfulness

# SANITY_CHECKS for attributions
results/sanity_checks: scripts/sanity_checks.py scripts/fig_sanity_checks.py
	# heat-rejection
	# data randomization:
	python3 scripts/train.py \
		--task heat-rejection \
		--data-dir data/task-heat-rejection \
		--num-runs 1 \
		--num-folds 1 \
		--num-epochs 1000 \
		--stopping-grace 1000 \
		--model-config results/hyperopt/task-heat-rejection/best_model_config.json \
		--log-dir results/models/randomized_labels \
		--run-group-name task-heat-rejection_randomized-labels-fit \
		--permute-labels True
	python3 scripts/interpret.py
		--task heat-rejection \
		--fitted-model-dir results/models/randomized_labels/task-heat-rejection_randomized-labels-fit \
		--data-dir data/task-heat-rejection \
		--attributions-dir results/attributions/randomized_labels/task-heat-rejection
	# model randomization:
	python3 scripts/interpret.py
		--task heat-rejection \
		--fitted-model-dir results/models/task-heat-rejection \
		--data-dir data/task-heat-rejection \
		--attributions-dir results/attributions/randomized_model/task-heat-rejection \
		--use-random-init True
	# sanity checks:
	python3 scripts/sanity_checks.py \
		--task heat-rejection \
		--data-dir data/task-heat-rejection \
		--attributions-dir results/attributions/task-heat-rejection \
		--randomized-labels-attributions-dir results/attributions/randomized_labels/task-heat-rejection \
		--randomized-model-attributions-dir results/attributions/randomized_model/task-heat-rejection \
		--sanity-checks-dir results/sanity_checks/task-heat-rejection

	# WM
	python3 scripts/train.py \
		--task WM \
		--data-dir data/task-WM \
		--num-runs 1 \
		--num-folds 1 \
		--num-epochs 1000 \
		--stopping-grace 1000 \
		--model-config results/hyperopt/task-WM/best_model_config.json \
		--log-dir results/models/randomized_labels \
		--run-group-name task-WM_randomized-labels-fit \
		--permute-labels True
	python3 scripts/interpret.py
		--task WM \
		--fitted-model-dir results/models/randomized_labels/task-WM_randomized-labels-fit \
		--data-dir data/task-WM \
		--attributions-dir results/attributions/randomized_labels/task-WM
	# model randomization:
	python3 scripts/interpret.py
		--task WM \
		--fitted-model-dir results/models/task-WM \
		--data-dir data/task-WM \
		--attributions-dir results/attributions/randomized_model/task-WM \
		--use-random-init True
	# sanity checks:
	python3 scripts/sanity_checks.py \
		--task WM \
		--data-dir data/task-WM \
		--attributions-dir results/attributions/task-WM \
		--randomized-labels-attributions-dir results/attributions/randomized_labels/task-WM \
		--randomized-model-attributions-dir results/attributions/randomized_model/task-WM \
		--sanity-checks-dir results/sanity_checks/task-WM

	# MOTOR
	python3 scripts/train.py \
		--task MOTOR \
		--data-dir data/task-MOTOR \
		--num-runs 1 \
		--num-folds 1 \
		--num-epochs 1000 \
		--stopping-grace 1000 \
		--model-config results/hyperopt/task-MOTOR/best_model_config.json \
		--log-dir results/models/randomized_labels \
		--run-group-name task-MOTOR_randomized-labels-fit \
		--permute-labels True
	python3 scripts/interpret.py
		--task MOTOR \
		--fitted-model-dir results/models/randomized_labels/task-MOTOR_randomized-labels-fit \
		--data-dir data/task-MOTOR \
		--attributions-dir results/attributions/randomized_labels/task-MOTOR
	# model randomization:
	python3 scripts/interpret.py
		--task MOTOR \
		--fitted-model-dir results/models/task-MOTOR \
		--data-dir data/task-MOTOR \
		--attributions-dir results/attributions/randomized_model/task-MOTOR \
		--use-random-init True
	# sanity checks:
	python3 scripts/sanity_checks.py \
		--task MOTOR \
		--data-dir data/task-MOTOR \
		--attributions-dir results/attributions/task-MOTOR \
		--randomized-labels-attributions-dir results/attributions/randomized_labels/task-MOTOR \
		--randomized-model-attributions-dir results/attributions/randomized_model/task-MOTOR \
		--sanity-checks-dir results/sanity_checks/task-MOTOR

	# Figure
	python3 scripts/fig_sanity_checks.py \
		--sanity-checks-base-dir results/sanity_checks
