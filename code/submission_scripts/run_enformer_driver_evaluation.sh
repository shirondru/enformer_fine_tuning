eval "$(/pollard/home/sdrusinsky/miniforge3/bin/conda shell.bash hook)"
source /pollard/home/sdrusinsky/miniforge3/bin/activate test_pt231

cd "$(dirname "${BASH_SOURCE[0]}")" #cd into the directory containing this script
cd .. #cd into `code` directory

script_path=./select_drivers_enformer.py
driver_method=forward_selection_with_only_drivers
select_drivers=false
evaluate_drivers=true
skip_finished_runs=false

python $script_path --driver_method $driver_method --select_drivers $select_drivers --evaluate_drivers $evaluate_drivers --skip_finished_runs $skip_finished_runs