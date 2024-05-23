# Project Title

## Description
This project involves testing different algorithms in kernel-based reward-free reinforcement learning (RL). The following options are available for configuration:

### Kernel Options for Transition Probability (`--P_kernel`)
- `RBF`
- `Matern_smoothness_1.5`
- `Matern_smoothness_2.5`

### Model Options (`--model`)
- `Benchmark_Model`
- `Greedy_Max_Variance`
- `without_generative_model`
- `Generative_model`

### Parameters
- `--beta`: Upper Confidence Bound (UCB) coefficient
- `--num_runs`: Number of sequential runs
- `--alpha`: Regularization hyperparameter for Gaussian Process (GP) regression used to generate the reward function \( r \) and transition probability \( P \).


### Installing Required Libraries
To install the required libraries, you can use:

pip install -r requirements.txt

### Rewards and P Folder

If you run `test_all_algos.py` for the first time, it will draw the reward function \( r \) and the transition probability \( P \) in the Reproducing Kernel Hilbert Space (RKHS) and store them in the `Rewards_and_P` folder. For subsequent runs, it will load these functions from this folder.

An example folder, `Rewards_and_P_experiment1`, is provided along with the results of the tested algorithms in the `experiment1` folder. The `final_plots.py` script generates graphs based on the data stored in the `experiment1` folder. Feel free to modify the paths in the script to plot your new experiments.


## Examples
Here are examples of how to run the code:

```sh
python test_all_algos.py --P_kernel RBF --beta 0.1 --num_runs 80 --model Benchmark_Model --alpha 0.01
python test_all_algos.py --P_kernel Matern_smoothness_1.5 --beta 0.1 --num_runs 80 --model Greedy_Max_Variance --alpha 0.5
python test_all_algos.py --P_kernel Matern_smoothness_2.5 --beta 0.1 --num_runs 80 --model without_generative_model --alpha 0.5
python test_all_algos.py --P_kernel Matern_smoothness_2.5 --beta 0.1 --num_runs 80 --model Generative_model --alpha 0.5
