# Machine Learning of the Phase Function

Summer-intern project supervised by Prof. [Ge Wang](https://biotech.rpi.edu/centers/bic/people/faculty/ge-wang), Rensselaer Polytechnic Institute (RPI).

Started from May, 2021.

## Objective

Building a phase function model for light propagation in medium using machine learning methods. 

Prof. Ge Wang: *We can “invert” M-C simulation (which can be CW, frequency-modulated, or time-resolved) to fit the underlying phase function for a homogeneous slab tissue: 1) simulate with HG phase functions with various parameters, and then learn from data so that we have a HG phase function equivalent network model, and 2) in MOSE we perturb the HG phase function and produce simulated data again, and then the network model can fit into the perturbed phase function.*  

## Background

### Henyey and Greenstein (H-G) phase function 

Reference of H-G function: [HG_Note](https://www.astro.umd.edu/~jph/HG_note.pdf)

Steven Jacques, Scott Prahl, [Introduction to Biomedical Optics](https://omlc.org/classroom/ece532/), Course of Oregon Graduate Institute

### Molecular Optical Simulation Environment (MOSE)

[MOSE](http://www.radiomics.net.cn/platform/docs/4) is a software platform for the simulation of light propagation in turbid media. The development of MOSE was initiated by Prof. Ge Wang and his collaborators. In this project, I use a modified command line version of MOSE provided by Dr. Shenghan Ren, the former team leader of MOSE. 

References:

1. Shenghan Ren, Xueli Chen, Hailong Wang, Xiaochao Qu, Ge Wang, Jimin Liang, and Jie Tian, [Molecular Optical Simulation Environment (MOSE): A Platform for the Simulation of Light Propagation in Turbid Media](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0061304), Plos One, 8(4):e61304, 2013.
2. Shenghan Ren, A Study on Efficient Method for Optical Tomography, Ph.D thesis, Xidian University, 2016. (In Chinese)

## Methods

### Development Environment 
    - Windows 10
    - Cuda 10.2
    - Matlab 2020b
    - Anaconda3 2021.05
    - Python 3.8
    - Pytorch 1.8.1

### Data Generation

The command line MOSE program is located in /MOSE along with the dependency runtime libraries. The phantom and simulation parameters are listed in the "mse" config file. Some parameters, such as the absorption coefficient $\mu_a$, scattering coefficient $\mu_s$, anisotropy factor $g$ of H-G function, and refractive index $n$, can be overridden by command line parameters. As a first attempt, the continuous wave mode of MOSE is used in the following experiments.

`` moseVCTest.exe configFileName outputDataFileName mu_a mu_s g n``

I wrote a MATLAB program ([Step1_GenerateRawData_CW.m](Step1_GenerateRawData_CW.m)) to simulate light propagation in a homogeneous slab tissue with typical optical parameters of $u_a=0.05$, $u_s=10$, and $n=1.3$. The anisotropy factor $g$ varies from -1 to 1 with a step of 0.025. For each g value, MOSE was run 100 times to generate train data for the neural network, and 30 times to generate the test data. 

Since MOSE saves all side-view observations of the slab, another MATLAB program ([Step2_ChangeRawData2Mat_CW.m](Step2_ChangeRawData2Mat_CW.m)) was written to extract only the top-view observations. The results are saved in mat format to maintain the data accuracy. 
