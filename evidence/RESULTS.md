# Real Experiment Results

This evidence pack consolidates outputs from completed prompting, retrieval and LoRA workflow runs.

## Real Hugging Face prompting

- fixed_few_shot: exact match=0.2500, token F1=0.2500
- retrieved_few_shot: exact match=0.2500, token F1=0.2500
- zero_shot: exact match=0.2500, token F1=0.2500

### Generated outputs

- e1 / zero_shot: expected='decrease', output='increase'
- e1 / fixed_few_shot: expected='decrease', output='increase'
- e1 / retrieved_few_shot: expected='decrease', output='increase'
- e2 / zero_shot: expected='hello', output='hELLO'
- e2 / fixed_few_shot: expected='hello', output='hELLO'
- e2 / retrieved_few_shot: expected='hello', output='hELLO'
- e3 / zero_shot: expected='5', output='Two plus three.'
- e3 / fixed_few_shot: expected='5', output='0'
- e3 / retrieved_few_shot: expected='5', output='3'
- e4 / zero_shot: expected='batteries', output='battery is a term used to refer to a unit of energy.'
- e4 / fixed_few_shot: expected='batteries', output='battery'
- e4 / retrieved_few_shot: expected='batteries', output='battery'

## Evidence files

- tables/phase1_prompting_baseline.csv
- tables/phase2_retrieval_prompting.csv
- tables/phase2_real_prompting.csv
- tables/phase3_base_vs_lora.csv
- tables/phase4_error_summary.csv
- lora_training_metrics.json
- final_project_summary.json

Large adapter/checkpoint files remain excluded from Git.
