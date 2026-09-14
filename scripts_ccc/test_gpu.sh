#!/bin/bash
#MSUB -q a100    
#MSUB -n 1      
#MSUB -T 3600
#MSUB -c 4

python test_gpu.py