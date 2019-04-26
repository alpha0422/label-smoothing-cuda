from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='label_smoothing',
    version='0.1',
    description='High performance PyTorch CUDA label smoothing.',
    packages=find_packages(), 
    ext_modules=[
        CUDAExtension('label_smoothing_cuda', [
            'csrc/label_smoothing_cuda.cpp',
            'csrc/label_smoothing_cuda_kernel.cu',
        ],
        extra_compile_args={
            'cxx': ['-O2',],
            'nvcc':['--gpu-architecture=sm_70']
        })
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
