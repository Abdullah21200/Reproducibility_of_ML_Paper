# Reproducibility of Research paper within Machine learning
Test the claims of the Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets paper

# Unsupervised Classification with SimCLR and SCAN

This repository contains code and resources for unsupervised classification tasks, specifically implementing **SimCLR** and **SCAN** on CIFAR-10.

We firstly do the **pretext task** (or use pretrained weights from SCAN repo) and run *SCAN* algorithm for different number of clusters

---

## Environment Setup

we used anaconda to install required packages.
To set up the required environment, use the provided `environment.yml` file.

```bash  
conda env create -f environment.yml
```
## Task 1: SimCLR
This task involves using a pre-trained SimCLR model and computing neighbors.

follow the scan repository (specifically the [TUTORIAL.MD](https://github.com/wvangansbeke/Unsupervised-Classification/blob/master/TUTORIAL.md) file)

## Task 2: SCAN

Slight edit here where we save the embeddings, cluster probabilites and can define the number of classes you want.

The steps should be the same in the **TUTORIAL.MD** file except for defining the number of clusters

```bash
python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml --num_clusters 10
python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml --num_clusters 20
python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml --num_clusters 30
python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml --num_clusters 40
python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml --num_clusters 50
python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml --num_clusters 60
```

This step will take a very long time.

## Task 3: Running the typicality formula

now if everything is stored and saved correctly (you may need to adjust paths), you can now run the notebook step by step.
should include the typicality formula from the paper, the random baseline and our cosine medoid modification for the fully supervised strategy. 



