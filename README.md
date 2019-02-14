# Deep Convolutional Sum-Product Networks (DCSPNs)

## About

We give conditions under which *convolutional neural networks* (CNNs) define valid *sum-product networks* (SPNs). One subclass, called *convolutional* SPNs (CSPNs), can be implemented using tensors, but also can suffer from being too shallow. Fortunately, tensors can be augmented while maintaining valid SPNs. This yields a larger subclass of CNNs, which we call *deep convolutional* SPNs (DCSPNs), where the convolutional and sum-pooling layers form rich directed acyclic graph structures. One salient feature of DCSPNs is that they are a rigorous probabilistic model. As such, they can exploit multiple kinds of probabilistic reasoning, including *marginal* inference and *most probable explanation* (MPE) inference. This allows an alternative method for learning DCSPNs using vectorized differentiable MPE, which plays a similar role to the generator in *generative adversarial networks* (GANs). Image sampling is yet another application demonstrating the robustness of DCSPNs. Our preliminary results on image sampling are encouraging, since the DCSPN sampled images exhibit variability. Experiments on image completion show that DCSPNs significantly outperform competing methods by achieving several state-of-the-art *mean squared error* (MSE) scores in both left-completion and bottom-completion in benchmark datasets.

## Examples

Small example for sanity test:

```bash
python run_experiment.py --name example --seed 1234 --width 64 --height 64 --channels 1 --output-dir outputs --database-path databases/olivetti --learning-rate 0.01 --minibatch-size 64 --epochs 3 --valid-amount 50 --first-sum-channels 12 --model-type tree --tree-model-size 16 --tree-model-alt-size 32 --tree-model-alt-amt 100 --leaf-components 4 --complete-side left --training-type nll --inference-type mpe
```

One of the paper experiments:

```bash
python run_experiment.py --name olivetti_left --seed 1234 --width 64 --height 64 --channels 1 --output-dir outputs --database-path databases/olivetti --learning-rate 0.01 --minibatch-size 64 --epochs 200 --valid-amount 50 --first-sum-channels 12 --model-type tree --tree-model-size 2 --tree-model-alt-size 2 --tree-model-alt-amt 100 --leaf-components 4 --complete-side left --training-type nll --inference-type mpe
```

## Reference
C.J. Butz, J.S. Oliveira, A. dos Santos, A.L. Teixeira, Deep Convolutional Sum-Product Networks, Thirty-Third AAAI Conference on Artificial Intelligence (AAAI), 2019.
