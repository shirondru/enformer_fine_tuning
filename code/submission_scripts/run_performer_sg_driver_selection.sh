eval "$(/pollard/home/sdrusinsky/miniforge3/bin/conda shell.bash hook)"
source /pollard/home/sdrusinsky/miniforge3/bin/activate test_pt231

cd "$(dirname "${BASH_SOURCE[0]}")" #cd into the directory containing this script
cd .. #cd into `code` directory

path_to_metadata=./metadata_from_past_runs/wandb_export_FinalPaperWholeBlood_SingleGene.csv
model_type=SingleGene
script_path=./select_drivers_performer.py
plot_selection=drivers
for driver_method in "forward_selection_with_only_drivers" "forward_selection"; do
    python $script_path --driver_method $driver_method --model_type $model_type --plot_selection $plot_selection --path_to_metadata $path_to_metadata
done