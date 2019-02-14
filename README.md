Deep Convolutional Sum-Product Networks (DCSPNs)
------------------------------------------------

# About
DCSPN code from [1].

# Examples

Small example for sanity test:

```
python run_experiment.py --name example --seed 1234 --width 64 --height 64 --channels 1 --output-dir outputs --database-path databases/olivetti --learning-rate 0.01 --minibatch-size 64 --epochs 3 --valid-amount 50 --first-sum-channels 12 --model-type tree --tree-model-size 16 --tree-model-alt-size 32 --tree-model-alt-amt 100 --leaf-components 4 --complete-side left --training-type nll --inference-type mpe
```

One of the paper experiments:

```
python run_experiment.py --name olivetti_left --seed 1234 --width 64 --height 64 --channels 1 --output-dir outputs --database-path databases/olivetti --learning-rate 0.01 --minibatch-size 64 --epochs 200 --valid-amount 50 --first-sum-channels 12 --model-type tree --tree-model-size 2 --tree-model-alt-size 2 --tree-model-alt-amt 100 --leaf-components 4 --complete-side left --training-type nll --inference-type mpe
```

# Reference
[1] C.J. Butz, J.S. Oliveira, A. dos Santos, A.L. Teixeira, Deep Convolutional Sum-Product Networks, Thirty-Third AAAI Conference on Artificial Intelligence (AAAI), 2019.
