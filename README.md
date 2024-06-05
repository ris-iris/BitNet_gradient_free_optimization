# BitNet gradient-free optimization

Implementation of several gradient-free optimization methods for BitNet - an extremely quantized Transformer model.

Project organization:
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
