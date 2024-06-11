# Torchpoly

Torchpoly is a PyTorch implementation of the paper ["An abstract domain for certifying neural networks"](https://dl.acm.org/doi/10.1145/3290354).

## Benchmarks

We tested the implementation on a few benchmarks. The results are as follows:

|        Benchmark      | Number of Holds | Number of Unkowns | Number of Fails |
|-----------------------|-----------------|-------------------|-----------------|
|    vnncomp23.acasxu   |        0        |         187       |         0       |
| vnncomp23.dist_shift  |        49       |         23        |         0       |
|vnncomp22.rl_benchmarks|        86       |         207       |         3       |
|     vnncomp21.eran    |        17       |         55        |         0       |
|   vnncomp21.mnistfc   |        0        |         90        |         0       |