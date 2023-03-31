# Homework 2 Report

## Question 1

The figures below show the average eval return through 100 iterations for batch sizes 1000 and 5000 across three variations of policy gradient:
1. Basic policy gradient, which at every timestep uses the sum of the rewards over all timesteps.
2. Policy gradient using reward-to-go.
3. Policy gradient using reward-to-go where we also standardise the advantages before computing the gradient.

| ![Learning curve for small batches](images/q1_small_batch.png) |
|:--:| 
| *Learning curve for small batches* |

| ![Learning curve for large batches](images/q1_large_batch.png) |
|:--:| 
| *Learning curve for large batches* |

Conclusions:
- With the smaller batch size of 1000, there's quite a lot of instability, so it's hard to draw strong conclusions.
- Using reward-to-go does however appear to improve performance, both in terms of how quickly the optimal return is approached, and how stable the training is.
- It's unclear if standardising the rewards leads to an improvement in either of these.
- Using the larger batch size of 5000 improves performance considerably. The training converges faster and is more stable. All variations converge to the optimum return by 100 iterations.


## Question 2

The figures below show the average proportion the training run which on which we receive the optimal return, for different learning rates and batch sizes, which the average taken across four random seeds. The second figure differs from the first in that we replace all points whose average proportion is less than 0.1 by an empty circle. 

| ![Average proportion of training run attaining the optimal return](images/q2_no_threshold.png) |
|:--:| 
| *Average proportion of training run attaining the optimal return* |

| ![Average proportion of training run attaining the optimal return, thresholding at 0.1](images/q2_0.1_threshold.png) |
|:--:| 
| *Average proportion of training run attaining the optimal return, thresholding at 0.1* |

Conclusions:
- The best set of parameters from those tested is a learning rate of 0.01 and a batch size of 3000.
- The highest learning rate and lowest batch size which gets above the 0.1 threshold is either (0.08, 900) or (0.09, 4000).