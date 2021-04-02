# AuTO

**AuTO**: A Framework for **Au**tomatic differentiation in **T**opology **O**ptimization

Aaditya Chandrasekhar, Saketh Sridhara, Krishnan Suresh  
[Engineering Representations and Simulation Lab](https://ersl.wisc.edu)  
University of Wisconsin-Madison  
The relevant paper can be found [here](https://ersl.wisc.edu/publications/).

## Abstract
A critical step in topology optimization (TO) is finding sensitivities. Manual derivation and implementation of the sensitivities can be quite laborious and error-prone, especially for non-trivial objectives, constraints and material models. An alternate approach is to utilize automatic differentiation (AD). While AD has been conceptualized over decades, and has also been applied in TO, wider adoption has largely been absent.

In this educational paper, we aim to reintroduce AD for TO and make it easily accessible through illustrative codes. In particular, we employ JAX, a high-performance Python library for automatically computing sensitivities from a user defined TO problem. The resulting framework, referred to here as AuTO, is illustrated through several examples in compliance minimization, compliant mechanism design and microstructural design.

## Code
We present four examples in TO (structural compliance, thermal compliance, design of compliant mechanisms and microstructural design). The standalone notebooks can be run using Google Colab or by installing the required dependencies (JAX, numpy, matplotlib, scipy).



