# AuTO

[**AuTO**: A Framework for **Au**tomatic differentiation in **T**opology **O**ptimization](https://ersl.wisc.edu/publications/2021/AuTO.pdf)

Aaditya Chandrasekhar, Saketh Sridhara, Krishnan Suresh  
[Engineering Representations and Simulation Lab](https://ersl.wisc.edu)  
University of Wisconsin-Madison 

## Abstract
A critical step in topology optimization (TO) is finding sensitivities. Manual derivation and implementation of the sensitivities can be quite laborious and error-prone, especially for non-trivial objectives, constraints and material models. An alternate approach is to utilize automatic differentiation (AD). While AD has been conceptualized over decades, and has also been applied in TO, wider adoption has largely been absent.

In this educational paper, we aim to reintroduce AD for TO and make it easily accessible through illustrative codes. In particular, we employ JAX, a high-performance Python library for automatically computing sensitivities from a user defined TO problem. The resulting framework, referred to here as AuTO, is illustrated through several examples in compliance minimization, compliant mechanism design and microstructural design.

## Code
The code can be run in two ways:
1. Using Google Colab
- Click on the file you want to run (for eg. compliance.ipynb)
- Click on "Open in Colab"
- Go to "Runtime" and select "Run All"
2. Clone and run
- install required dependencies (JAX, numpy, matplotlib, scipy).
- For windows machines: use WSL
