# RECOVER
The code for the paper "An Online Method for A Class of Distributionally Robust Optimization with Non-Convex Objectives" [![pdf](https://img.shields.io/badge/Arxiv-pdf-orange.svg?style=flat)](https://arxiv.org/pdf/2006.10138.pdf)


Configuration
---------
pip3 install torch==1.4.0 torchvision==0.5.0 --user


Implementation
---------
Let \x represnets the sample. To taking the most advantage of GPU data-parallel for RECOVER implementation, with two GPUs, the first GPU stores data batch {\x_t} and second GPUs store data batch {\x_{t+1}}.
And in our implementation, we take two steps for one u, v updates.
Take u for example
```python
 u_{t+1} = g(\w_{t+1}, \x_{t+1}) + (1-a_{t+1})*(u_{t} - g(\w_{t},x_{t+1}))
``` 

In our implementation, u_t = y_t, loss1_max and loss2_max are for algorithm stablization.


```python
# For the current iteration t, in the second GPU
# updates  u_t =  u_{t} - g(\w_{t},x_{t+1})
args.y_t = args.y_t - exp_loss_2.item() * torch.exp(loss2_max / args.curlamda)
# Then in the next loop, in the first GPU
# g = exp(\ell(\w_{t+1}, \x_{t+1})/\lambda)
# u_{t+1} = g(\w_{t+1}, \x_{t+1}) +  (1-a_t)*u_t 
args.y_t = exp_loss_1 * torch.exp(
             loss1_max / args.curlamda) + (1-args.a_t) * args.y_t
```

These two steps together finish the updates of u.
Similar updates implementation design also for v.



To replicate the results:
---------
```python
bash run_RECOVER.sh
```

Citation
---------
If you find this repo helpful, please cite the following paper:
```
@article{qi2021online,
  title={An online method for a class of distributionally robust optimization with non-convex objectives},
  author={Qi, Qi and Guo, Zhishuai and Xu, Yi and Jin, Rong and Yang, Tianbao},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

Contact
----------
If you have any questions, please contact us @ [Qi Qi](https://qiqi-helloworld.github.io/) [qi-qi@uiowa.edu] , and [Tianbao Yang](https://homepage.cs.uiowa.edu/~tyng/) [tianbao-yang@uiowa.edu] or please open a new issue in the Github. 
