# FLOWR: Flow Matching for Structure-Aware De Novo, Interaction- and Fragment-Based Ligand Generation

FLOWR is a research repository that investigates continuous and discrete flow matching methods applied to structure-based drug discovery. It provides a complete workflow for training models, generating novel ligand molecules, and evaluating the generated structures.

---

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
  - [Data](#data)
  - [Training the Model](#training-the-model)
  - [Generating Molecules](#generating-molecules)
  - [Evaluating Molecules](#evaluating-molecules)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Installation

1. **Create the Environment**  
   Install the required environment using [mamba](https://mamba.readthedocs.io):

   ```bash
   mamba env create -f environment.yml
   ```

2. **Activate the Environment**  

   ```bash
   conda activate flowr
   ```

3. **Set PYTHONPATH**  
   Ensure the repository directory is in your Python path:

   ```bash
   export PYTHONPATH="$PWD"
   ```

---

## Getting Started

We provide the full SPINDR data in both .smol and .cif format, as well as a fully trained FLOWR model checkpoint and generated samples.
For training, generation and evaluation, we provide basic bash and SLURM scripts in the `scripts/` directory. These scripts are intended to be modified and adjusted according to your computational resources and experimental needs.

### Data
Download the SPINDR dataset, the FLOWR checkpoint and generated samples here:
[Zenodo](https://zenodo.org/uploads/15212510)
To train a model, untar the smol_data.tar to get the smol-files. Specify the directory they are placed in the respective scripts (see below).
We also provide the cif-files for all protein pockets splitted into train, validation and test.
For running generation only, place the flowr.ckpt path somewhere and specify its location in the `scripts/gen_spindr.sl` script.

### Training the Model

Start by training the model using the provided training script. This script sets hyperparameters such as batch size, learning rate, and network architecture.

Modify `scripts/train_spindr.sh` as needed, then run:

```bash
bash scripts/train_spindr.sh
```

### Generating Molecules

After training, generate novel molecules using the generation script. This script supports GPU execution and includes options such as sampling steps and noise injection.

Modify `scripts/gen_spindr.sl` according to your requirements, then submit the job via SLURM:

```bash
sbatch scripts/gen_spindr.sl
```

### Evaluating Molecules

Evaluate the generated molecules using the evaluation script. This step calculates metrics including molecular validity, uniqueness, and interaction recovery.

Modify `scripts/eval_spindr.sh` as needed, then run:

```bash
bash scripts/eval_spindr.sh
```

---

## Contributing

Contributions are welcome! If you have ideas, bug fixes, or improvements, please open an issue or submit a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Citation

If you use FLOWR in your research, please cite it as follows:

```bibtex
@article{flowr_2025,
  title={FLOWR – Flow Matching For Structure-Aware De Novo,
Interaction- and Fragment-Based Ligand Generation},
  author={Julian Cremer, Ross Irwin et al.},
  journal={arxiv},
  year={2025},
  doi={DOI}
}
```

---