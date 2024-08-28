# BitNet gradient-free optimization

Implementation of several gradient-free optimization methods for BitNet - an extremely quantized Transformer model.

## Results
Please refer to the [project report](report/report.pdf)

## Project organization
<table>
    <thead>
        <tr>
            <th>directory</th>
            <th>filename</th>
            <th>functionalty</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>model</td>
            <td> bit_linear.py </td>
            <td> Quantized linear layer </td>
        </tr>
        <tr>
            <td> transformer.py </td>
            <td> Transformer with bit_linear layers </td>
        </tr>
        <tr>
            <td> sa.py </td>
            <td> Transformer based classifier for sentiment analysis</td>
        </tr>
        <tr>
            <td rowspan=4>optim</td>
            <td>adam.py </td>
            <td> Baseline adam optimizer </td>
        </tr>
        <tr>
            <td> mcmc.py</td>
            <td> Markov chain Monte Carlo </td>
        </tr>
        <tr>
            <td> simple_ga.py</td>
            <td> Mutation only genetic algorithm </td>
        </tr>
        <tr>
            <td> simulated_annealing.py</td>
            <td> Simulated annealing </td>
        </tr>
        <tr>
            <td rowspan=3>datasets</td>
            <td>addition.py </td>
            <td> Addition of 3 numbers equations </td>
        </tr>
        <tr>
            <td> generate_brackets_dataset.py, brackets.py </td>
            <td> Correct and incorrect bracket sequences classification </td>
        </tr>
        <tr>
            <td> generate_twitter_sa_dataset.py, sa.py </td>
            <td> Twits sentiment analysis </td>
        </tr>
    </tbody>
</table>

## Usage

1. Install dependencies
```
pip install -r requirements.txt
```
2. Login to wandb
```
wandb login
```
3. Look at all the argumetns you can specify (copied main.py)

| argument | possible values | description|
|----------|-----------------|------------|
|--dataset | "twitter", "addition", "brackets" | The dataset to use for training and testing. |
|--batch_size | int | The batch size for training. |
|--epochs | int | The number of epochs to train for. |
|--optimizer | "adam", "simple_ga", "mcmc", "sim_annealing", "zeroth" | The optimizer to use for training. |
|--seed | int | The random seed for reproducibility. |
|--model | "bit_transformer", "bit_sa_transformer" | The model architecture to use for training. |
|--max_length | int | The maximum sequence length for the model. |
|--data_repo | "./data/" | The directory where the dataset is stored. |
|--track_ops | bool | The flag that specifies if operations will be tracked. |
|--lr | float | Adam Optimizer parameter: learning rate |
|--beta1 |  float | Adam Optimizer parameter: beta1|
|--beta2 | float | Adam Optimizer parameter: beta2 |
|--weight_decay | float | Adam Optimizer parameter: weight decay |
|--warmup_steps | int | Adam Optimizer parameter: warmup steps|
|--max_grad_norm | int | Adam Optimizer parameter: max grad norm|
|--population_size | int | Genetic Optimizer parameter: population size |
|--treshold | int | Genetic Optimizer parameter: selection treshold |
|--bin_mutation_prob | float | Genetic or MCMC Optimizer parameter: probability of mutating binary value |
|--emb_mutation_scale | float | Genetic or MCMC Optimizer parameter: mutation scale for notmal mutaion of embeddings |
|--initial_temp | float | Simulated annealing Optimizer parameter: initial temperature |
|--cooling_rate | float | Simulated annealing Optimizer parameter: cooling rate |
|--min_temp | float | Simulated annealing Optimizer parameter: min temperature |
|--random_vec | int | Zeroth-order Optimizer parameter: number of random vectors for estimation |
|--momentum | float | Zeroth-order Optimizer parameter: momentum |
|--grad_mode | "zeroth_order_rge", "zeroth_order_cge" | Zeroth-order Optimizer parameter: algorithm to use|

4. Run main.py with parameters of interest

```
python main.py --dataset twitter --optimizer simple_ga
```

### Hyperparameter tuning

If you are interested in finding optimal hyperparameters for some algorithm, you can use `wandb sweep` to do it. We define several sweep configurations in the directory `hyperparams_opt`. To start sweep do:

```
wandb sweep hyperparams_opt/<config of interest>.yaml
wandb agent --count <number of runs to try> <sweep_id>
```
