### Vast AI
### check public ssh
> .ssh % cat ~/.ssh/vsai.pub

### connect the instance from the terminal

> ssh -p 43279 root@93.99.191.159 -L 8080:localhost:8080 -i ~/.ssh/vsai

### Upload files to vast ai
> scp -r -P 43279 -i ~/.ssh/vsai . root@93.99.191.159:/root/

### Download files from vast ai
> scp -r -P 43279 -i ~/.ssh/vsai root@93.99.191.159:/root/results .


## Environment setup
To create conda env we can run
> conda env create -f conda.yaml

To activate conda env
> conda activate python3.10-la-proj

To update conda env
> conda env update --name python3.10-la-proj --file conda.yaml --prune

To add the conda env to Jupyter as a new kernel
> python -m ipykernel install --user --name python3.10-la-proj --display-name "python3.10-la-proj"

## Install
> chmod +x install.sh
> ./install.sh 