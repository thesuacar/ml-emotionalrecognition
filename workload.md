# Emotion Recognition Assignment --- Team Checklist

> **How to use this file in VS Code** - Open this file in VS Code. -
> Tick off a task by changing `- [ ]` to `- [x]`. - Add names to
> **Owner** and dates to **Due** before starting. - Commit/push changes
> regularly so the whole team sees the same checklist. - Keep
> evidence/results in the repo and link them under the relevant task
> where useful.

## Project goal

Build and evaluate machine-learning models to detect **7 facial
emotions** across **FER2013** and **KDEF**. The work includes
cross-dataset/domain-shift evaluation, a fuzzy classifier, a creative
performance challenge, and a real-time webcam demo.

### Final deliverables

-   [ ] 8-page PDF report
-   [ ] Notebooks used for experiments
-   [ ] Saved `.pkl` model file(s)
-   [ ] `demo.py` webcam demo
-   [ ] `requirements.txt`
-   [ ] Final ZIP containing the required deliverables

### Team planning

Team members: **A:** Su Acar (2475901) **B:** \_\_\_\_\_\_\_\_\_\_
**C:** \_\_\_\_\_\_\_\_\_\_

Suggested starting split from the original plan (adjust as needed): -
Person A: Tasks **T04--T07** (baseline feature extraction, training,
evaluation) - Person B: Tasks **T08--T12** (KDEF, cross-dataset
evaluation, fuzzy classifier) - Person C: Tasks **T13--T17** (creative
challenge and demo) - Reporting: split **T18--T22** evenly across all
three people - T01--T03 and T23: agree an owner together

> Estimates below are the original task estimates, not guaranteed
> durations. Due dates and owners were not specified in the source plan.

------------------------------------------------------------------------

## Phase 1 --- Environment and data

### T01 · Set up the Python environment

-   [X] **Task:** Set up Python and required packages: `dlib`, `cv2`,
    `pyFUME`, `imbalanced-learn`.
-   **Owner:** Su **Due:** October 8
    **Estimate:** 15 min
-   **Done when:** `environment.yml` is pushed to the repository and
    all 3 team members can install it locally.
-   **Evidence / notes:**
    Done. Venv set up, repo created, pushed with no problems

### T02 · Split FER2013

-   [X] **Task:** Run the provided FER2013 splitting notebook.
-   **Owner:** Su Acar **Due:** October 8
    **Estimate:** 10 min
-   **Done when:** `fer2013_train.csv`, `fer2013_validation.csv`, and
    `fer2013_test.csv` have been generated.
-   **Evidence / notes:**
    download data yourself from kaggle. the csv files will drop in data/fer/

### T03 · Preprocess KDEF

-   [X] **Task:** Run the KDEF subject split and execute `preprocess.py`
    on KDEF images.
-   **Owner:** Su **Due:** October 8
    **Estimate:** 20 min
-   **Done when:** Frontal KDEF faces are cropped, converted to
    grayscale, and resized to `48x48`.
-   **Evidence / notes:**
    Unsure about correctness. may need to refer back to this step if an error occurs in training.
    Put data in data/kdef/

------------------------------------------------------------------------

## Phase 2 --- Task 1: Baseline models

### T04 · Build HOG and PCA feature extraction

-   [ ] **Task:** Implement HOG and PCA feature-extraction pipelines.
-   **Owner:** \_\_\_\_\_\_\_\_\_\_ **Due:** \_\_\_\_\_\_\_\_\_\_
    **Estimate:** 25 min
-   **Done when:** Functions convert `48x48` images into HOG and PCA
    feature arrays.
-   **Evidence / notes:**
    \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### T05 · Train Logistic Regression and SVM on FER2013

-   [ ] **Task:** Train Logistic Regression and SVM using the FER2013
    training split.
-   **Owner:** \_\_\_\_\_\_\_\_\_\_ **Due:** \_\_\_\_\_\_\_\_\_\_
    **Estimate:** 25 min
-   **Done when:** Both models are trained and their weights are
    temporarily saved.
-   **Evidence / notes:**
    \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### T06 · Train Random Forest baseline

-   [ ] **Task:** Train the FER2013 Random Forest baseline using the
    provided notebook.
-   **Owner:** \_\_\_\_\_\_\_\_\_\_ **Due:** \_\_\_\_\_\_\_\_\_\_
    **Estimate:** 20 min
-   **Done when:** `rf_fer_model_landmarks.pkl` is saved.
-   **Evidence / notes:**
    \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### T07 · Evaluate baseline models

-   [ ] **Task:** Evaluate Task 1 models on the FER2013 validation set.
-   **Owner:** \_\_\_\_\_\_\_\_\_\_ **Due:** \_\_\_\_\_\_\_\_\_\_
    **Estimate:** 20 min
-   **Done when:** Accuracy, balanced accuracy, Macro-F1, and confusion
    matrices are generated.
-   **Evidence / notes:**
    \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

------------------------------------------------------------------------

## Phase 3 --- Task 2: KDEF evaluation

### T08 · Extract the selected features on KDEF

-   [ ] **Task:** Use the best-performing features identified in T07 on
    KDEF images.
-   **Owner:** \_\_\_\_\_\_\_\_\_\_ **Due:** \_\_\_\_\_\_\_\_\_\_
    **Estimate:** 15 min
-   **Done when:** KDEF images are processed into the same feature space
    as FER2013.
-   **Evidence / notes:**
    \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### T09 · Train and evaluate on KDEF

-   [ ] **Task:** Retrain the best-performing classifiers on KDEF using
    an 80/20 subject split.
-   **Owner:** \_\_\_\_\_\_\_\_\_\_ **Due:** \_\_\_\_\_\_\_\_\_\_
    **Estimate:** 25 min
-   **Done when:** Subject-independent models are trained and evaluated,
    and metrics are generated.
-   **Evidence / notes:**
    \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

------------------------------------------------------------------------

## Phase 4 --- Task 3: Cross-dataset evaluation

### T10 · Test KDEF-trained model on FER2013

-   [ ] **Task:** Evaluate the KDEF-trained model on FER2013.
-   **Owner:** \_\_\_\_\_\_\_\_\_\_ **Due:** \_\_\_\_\_\_\_\_\_\_
    **Estimate:** 20 min
-   **Done when:** Domain-shift performance is evaluated and metrics
    plus confusion matrices are recorded.
-   **Evidence / notes:**
    \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

------------------------------------------------------------------------

## Phase 5 --- Task 4: Fuzzy classifier

### T11 · Extract geometric features

-   [ ] **Task:** Use `dlib` to extract 3--10 handcrafted geometric
    features, such as mouth width and eye opening.
-   **Owner:** \_\_\_\_\_\_\_\_\_\_ **Due:** \_\_\_\_\_\_\_\_\_\_
    **Estimate:** 30 min
-   **Done when:** A DataFrame of geometric features is generated for
    KDEF.
-   **Evidence / notes:**
    \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### T12 · Build and train the fuzzy classifier

-   [ ] **Task:** Scaffold and train a fuzzy classifier using `pyFUME`.
-   **Owner:** \_\_\_\_\_\_\_\_\_\_ **Due:** \_\_\_\_\_\_\_\_\_\_
    **Estimate:** 30 min
-   **Done when:** Human-understandable rules are generated and
    evaluated on KDEF.
-   **Evidence / notes:**
    \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

------------------------------------------------------------------------

## Phase 6 --- Task 5: Creative performance challenge

### T13 · Implement creative feature 1

-   [ ] **Task:** Implement one creative improvement, for example data
    augmentation or SMOTE.
-   **Owner:** \_\_\_\_\_\_\_\_\_\_ **Due:** \_\_\_\_\_\_\_\_\_\_
    **Estimate:** 25 min
-   **Done when:** The training pipeline uses a technique intended to
    handle data imbalance or variety.
-   **Evidence / notes:**
    \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### T14 · Implement creative feature 2

-   [ ] **Task:** Implement a second creative improvement, for example
    hyperparameter tuning or KDEF insights.
-   **Owner:** \_\_\_\_\_\_\_\_\_\_ **Due:** \_\_\_\_\_\_\_\_\_\_
    **Estimate:** 30 min
-   **Done when:** The best pipeline is optimized and a performance
    improvement is verified on FER2013.
-   **Evidence / notes:**
    \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### T15 · Finalize and save the production model

-   [ ] **Task:** Finalize the best model and save the production `.pkl`
    file.
-   **Owner:** \_\_\_\_\_\_\_\_\_\_ **Due:** \_\_\_\_\_\_\_\_\_\_
    **Estimate:** 15 min
-   **Done when:** The final model is evaluated on the private test
    split and the `.pkl` is saved to the repository.
-   **Evidence / notes:**
    \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

------------------------------------------------------------------------

## Phase 7 --- Task 6: Webcam demo

### T16 · Adapt the demo script

-   [ ] **Task:** Adapt `baseline_demo_fer2013.py` to use the new best
    model.
-   **Owner:** \_\_\_\_\_\_\_\_\_\_ **Due:** \_\_\_\_\_\_\_\_\_\_
    **Estimate:** 20 min
-   **Done when:** The script loads the new `.pkl` and handles real-time
    webcam frame extraction.
-   **Evidence / notes:**
    \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### T17 · Test the webcam demo

-   [ ] **Task:** Test the webcam demo locally and log performance
    issues.
-   **Owner:** \_\_\_\_\_\_\_\_\_\_ **Due:** \_\_\_\_\_\_\_\_\_\_
    **Estimate:** 15 min
-   **Done when:** The script runs without crashing, and success/failure
    cases are noted for the report.
-   **Evidence / notes:**
    \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

------------------------------------------------------------------------

## Phase 8 --- Report and final submission

### T18 · Write report: baseline and KDEF

-   [ ] **Task:** Draft Task 1 (Baseline) and Task 2 (KDEF) sections.
-   **Owner:** \_\_\_\_\_\_\_\_\_\_ **Due:** \_\_\_\_\_\_\_\_\_\_
    **Estimate:** 30 min
-   **Done when:** Feature/classifier ranking, difficult emotions, and
    dataset differences are described.
-   **Evidence / notes:**
    \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### T19 · Write report: cross-dataset and fuzzy classifier

-   [ ] **Task:** Draft Task 3 (Cross-Dataset) and Task 4 (Fuzzy)
    sections.
-   **Owner:** \_\_\_\_\_\_\_\_\_\_ **Due:** \_\_\_\_\_\_\_\_\_\_
    **Estimate:** 30 min
-   **Done when:** Domain shift is explained, and fuzzy rules are
    documented and justified.
-   **Evidence / notes:**
    \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### T20 · Write report: creative challenge and demo

-   [ ] **Task:** Draft Task 5 (Creative) and Task 6 (Demo) sections.
-   **Owner:** \_\_\_\_\_\_\_\_\_\_ **Due:** \_\_\_\_\_\_\_\_\_\_
    **Estimate:** 30 min
-   **Done when:** Creative choices are justified and webcam adaptation
    issues are described.
-   **Evidence / notes:**
    \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### T21 · Assemble confusion-matrix appendix

-   [ ] **Task:** Add all confusion matrices to the report appendix.
-   **Owner:** \_\_\_\_\_\_\_\_\_\_ **Due:** \_\_\_\_\_\_\_\_\_\_
    **Estimate:** 15 min
-   **Done when:** All matrices are included without excessive
    explanatory text.
-   **Evidence / notes:**
    \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### T22 · Finalize AI statement and report formatting

-   [ ] **Task:** Draft the AI usage statement and format the final
    report.
-   **Owner:** \_\_\_\_\_\_\_\_\_\_ **Due:** \_\_\_\_\_\_\_\_\_\_
    **Estimate:** 20 min
-   **Done when:** The official AI statement is completed and the PDF is
    no more than 8 pages.
-   **Evidence / notes:**
    \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### T23 · Package the final ZIP

-   [ ] **Task:** Package the final deliverables into a ZIP.
-   **Owner:** \_\_\_\_\_\_\_\_\_\_ **Due:** \_\_\_\_\_\_\_\_\_\_
    **Estimate:** 10 min
-   **Done when:** The ZIP contains the 8-page PDF, notebooks, `.pkl`
    models, and `demo.py`.
-   **Evidence / notes:**
    \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

------------------------------------------------------------------------

## Final team sign-off

-   [ ] All 23 tasks are checked off or explicitly agreed as not
    applicable.
-   [ ] The repository contains the latest notebooks, scripts,
    dependencies, and required model files.
-   [ ] The report is checked against the 8-page limit and includes the
    required sections and confusion matrices.
-   [ ] The webcam demo has been tested.
-   [ ] The final ZIP has been opened and its contents verified.
-   [ ] All team members agree the submission is ready.

**Final ZIP path:**
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Final report path:**
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Anything still blocked / unresolved:**\
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
