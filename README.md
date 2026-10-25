# Emotion recognition: FER2013 and KDEF

## Environments

Two conda environments are needed. The fuzzy classifier (Task 4) needs its own environment because PyFUME requires NumPy 1.24.4.

| Environment file | Used by | Key versions |
|---|---|---|
| `environments/environment-general.yml` | Notebooks 00a, 00b, 01, 02, 03, 05 | Python 3.12.15, scikit-learn 1.9.1, dlib 20.0.1 |
| `environments/environment-fuzzy.yml` | Notebook 04 (Task 4), notebook 06 and `demo.py` (Task 6) | Python 3.10.22, NumPy 1.24.4, PyFUME 0.3.4, dlib 19.24.2, scikit-learn 1.7.2 |

Create an environment with:

```bash
conda env create -f environments/environment-general.yml
conda env create -f environments/environment-fuzzy.yml
```

`requirements.txt` lists the pinned packages of the general environment, for use with `pip` in a Python 3.12 environment. The fuzzy environment must still be created from its YAML file.

## Running the notebooks

Run them in this order:

1. `00a_data_fer2013_split.ipynb` (general)
2. `00b_data_kdef_split.ipynb` (general)
3. `01_task1_fer2013_models.ipynb` (general)
4. `02_task2_kdef_dataset_study.ipynb` (general)
5. `03_task3_cross_dataset_generalization.ipynb` (general)
6. `04_task4_fuzzy_classifier.ipynb` (fuzzy)
7. `05_task5_creative_challenge.ipynb` (general)
8. `06_task6_webcam_demo.ipynb` (fuzzy kernel)

Notebook 01 uses `scripts/fer_features.py` and notebook 02 uses `scripts/feature_extraction.py`. Both are imported from `scripts/`. The dlib landmark model must be in `models/faceAndLandmarkModels/`.

## How to run the live demo

Make sure you're running from the fuzzy environment (`environments/environment-fuzzy.yml`), then run:

```bash
conda activate emotion_fuzzy
python demo.py
```
