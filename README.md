A high performance and minimal memory requirement CUDA implementation of label smoothing with cross entropy loss for PyTorch.

Notice
------

This repo has been merged with NVIDIA APEX at [apex/contrib/xentropy](https://github.com/NVIDIA/apex/tree/master/apex/contrib/xentropy), please use the APEX version instead. This repo will no longer be maintained.

How to use
----------

```console
$ git clone https://github.com/alpha0422/label-smoothing-cuda.git
$ cd label-smoothing-cuda
$ python setup.py install
$ python test/test.py
```

Performance
-----------

On DGX1V, we observed 5.62x ~ 6.87x performance improvement:

```console
# N, T, H = 32, 33, 32320
$ python test/test.py
Opt time 0.46 s elapsed for 1000 iterations, norm 0.83447265625
Raw time 3.16 s elapsed for 1000 iterations, norm 0.83447265625
Norm difference check passed!

# N, T, H = 128, 74, 32320
$ python test/test.py
Opt time 3.41 s elapsed for 1000 iterations, norm 0.62451171875 
Raw time 19.18 s elapsed for 1000 iterations, norm 0.62451171875
Norm difference check passed!
```

