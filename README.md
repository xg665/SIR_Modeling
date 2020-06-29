# SIR_Modeling
## Instruction for python initial setting after first login:
```
    ## Initial setting
    srun --nodes=1 --cpus-per-task=1 --time=1:00:00 --gres=gpu:1  --mem=16GB --pty /bin/bash
    module purge
    module load anaconda3/5.3.1
    module load cuda/10.0.130
    module load gcc/6.3.0

    ## Create the virtual environment and activate it
    NETID=mw3706
    mkdir /scratch/${NETID}/model
    mkdir /scratch/${NETID}/model/code
    conda create --prefix /scratch/${NETID}/model/env python=3.7
    source activate /scratch/${NETID}/model/env
    ## Customer python packages
    conda install numpy
    conda install matplotlib
    conda install networkx
```

## Upload test file and run it through script (Using slurm in hpc)
    - Open another terminal, use `scp` instruction. The format is `scp source destination`
        - First transfer the test file:
            ```
            scp /Users/oscar/Desktop/test.py mw3706@prince.hpc.nyu.edu:/scratch/mw3706/model/code
            ```
        - And also the script file
          ```
          scp /Users/oscar/Desktop/model_job.sh mw3706@prince.hpc.nyu.edu:/scratch/mw3706/model/code
          ```
    - Run the script by `sbatch`
        ```
        sbatch model_job.sh
        ```

## Some thoughts when I explore prince:
    - prince can also support matlab, and I think the startup is simpler than python (since it does not required
        much packages and environment)
    - prince also support GUI windows for jupyter notebook, so maybe it will be a bit more friendly if you are
        not familiar with terminal instructions. Here is the enter for jupyter access: https://jupyter.hpc.nyu.edu/
        But I didn't test functions on notebook, so I'm not sure whether it can be more convenient than terminal
