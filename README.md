# Feature-Distillation-for-3D-UDA

Official repository for "Self-Distillation for Unsupervised 3D Domain Adaptation" [Accepted at WACV2023]

[[Project page]](https://cvlab-unibo.github.io/FeatureDistillation/) [[Paper]](https://arxiv.org/abs/2210.08226)

### Authors

[Adriano Cardace](https://www.unibo.it/sitoweb/adriano.cardace2) - [Riccardo Spezialetti](https://www.unibo.it/sitoweb/riccardo.spezialetti) - [Pierluigi Zama Ramirez](https://pierlui92.github.io/) - [Samuele Salti](https://vision.deis.unibo.it/ssalti/) - [Luigi Di Stefano](https://www.unibo.it/sitoweb/luigi.distefano/)


## Requirements
We rely on several libraries: [Pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning), [Weight & Biases](https://docs.wandb.ai/), [Hesiod](https://github.com/lykius/hesiod)

```bash
conda env create -f environment.yml
conda activate self_distillation
```

## Download and load datasets on W&B server (Registration required)
Reqeuest dataset access at https://drive.google.com/file/d/14mNtQTPA-b9_qzfHiadUIc_RWvPxfGX_/view?usp=sharing.

The dataset is the same provided by the original authours at https://github.com/canqin001/PointDAN. For convenience we provide a preprocessed version used in this work.
Then, load all the required dataset to the wandb server:

```bash
mkdir data
unzip PointDA_aligned.zip -d data/
./load_data.sh
```

Please, set the variable in load_data.sh according to your wandb data, i.e. entity name and project name. 
You should create an experiment on wandb for each setting, for example "m2s" of "m2sc" ecc. Then, load only the required datasets for that experiment. 
In the provided script, we load "modelnet" and "scannet" to the "m2sc" project. 

## Training

To train modelnet->scannet, set the entity filed inside "main.py" in the wandb.init function and execute the following commands:

```bash
# step1 
python main.py logs/m2sc/step1/run.yaml
# generate pseudo labels and load data to wandb
python generate_pseudo_labels.py modelnet scannet
# step 2
python main.py logs/m2sc/step2/run.yaml
```

If you want to train on other settings you can rely on the run.yaml files stored in each folder experiment.
For each experiment, we also release the checkpoints. Note that results might be sligthly different from the paper as we run again all experiments once before releasing the code. 