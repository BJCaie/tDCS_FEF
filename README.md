# Variability in the Effects of Transcranial Direct Current Stimulation (tDCS) on Free Choice Behaviour

**Authors:** Brandon Caie¹* and Gunnar Blohm¹  
¹ Centre for Neuroscience Studies, Queen’s University  
*Corresponding Author:* [brandon.caie@queensu.ca](mailto:brandon.caie@queensu.ca)

---

## 🧠 Overview

This repository accompanies the paper:

> **Caie, B. & Blohm, G. (2025). _Variability in the effects of transcranial direct current stimulation on free choice behaviour._ Centre for Neuroscience Studies, Queen’s University.**

The study investigates **intra- and inter-individual variability** in how **high-definition transcranial direct current stimulation (HD-tDCS)** affects neural and behavioural measures during a **free-choice saccade task**.  
Across **10 sessions per participant (5 participants total)**, we combined behavioural, EEG, and fMRI data to evaluate how stimulation polarity (anodal vs cathodal) influenced:

- Reaction times and psychometric choice curves  
- EEG time–frequency responses  
- Variability within and across individuals

We developed a **difference-in-differences (DiD) permutation testing framework** to assess causal effects of tDCS at the **group, individual, and session levels**.

---

## 📊 Repository Contents

| Folder | Description |
|--------|--------------|
| `scripts/` | Scripts to run all analyses and plot paper results. |
| `figures/` | All figures and plots corresponding to results in the paper. |
| `notebooks/` | Jupyter notebooks for reproducing key analyses and figures. |


## ⚙️ Installation & Dependencies

- Clone the repository:
- git clone https://github.com/<username>/tDCS-variability.git
- cd tDCS-variability
- pip install -r requirements.txt

---

## Main Python Packages

* `numpy`, `pandas`, `scipy`
* `mne` (for EEG analysis)
* `psignifit` (for psychometric fitting)
* `matplotlib`, `seaborn` (for plotting)
* `nibabel`, `simnibs` (for current modeling)
* `pingouin` or `statsmodels` (for permutation & DiD analysis)

---

## 🧪 Experiment Summary

| Parameter | Description |
|------------|--------------|
| **Participants** | 5 healthy adults |
| **Sessions per participant** | 10 (5 anodal, 5 cathodal) over ~5 weeks |
| **Task** | Free-choice saccade task |
| **Modality** | Behavioural + EEG + fMRI (for FEF localization) |
| **Stimulation site** | Right Frontal Eye Field (rFEF) |
| **Stimulation type** | High-definition tDCS (HD-tDCS), 4×1 montage |
| **Current** | 2 mA center electrode, –0.5 mA per return electrode |
| **Stimulation duration** | 20 minutes (ramp-up/ramp-down: 30 s) |

---

### Task

Participants were presented with two peripheral visual targets (left/right).  
Each trial consisted of:

1. **Fixation phase:** central fixation for 750–1250 ms  
2. **Target onset:** both targets appear with a **temporal onset asynchrony (TOA)** ranging from –99 to +99 ms (left/right lead)  
3. **Response:** participants made a **saccadic eye movement** toward their chosen target  
4. **Feedback:** none (to preserve free-choice behaviour)

Trials were divided equally into **pre-tDCS** and **post-tDCS** blocks, allowing for within-session comparison.

### Stimulation Details

| Parameter | Value |
|------------|--------|
| **Montage type** | 4×1 HD-tDCS |
| **Centre electrode** | Anode or cathode (depending on condition) placed over rFEF |
| **Return electrodes** | Surrounding electrodes at 3.5 cm radius |
| **Localization method** | Individual fMRI used to determine rFEF coordinates |
| **Simulation tool** | *SimNIBS* for individualized electric field modeling |

Each participant received **both anodal and cathodal** stimulation across sessions in a randomized order to mitigate order and adaptation effects.

---

## ⏱️ Data Collected

| Data Type | Description |
|------------|-------------|
| **Behavioural** | Reaction times (RT), choice (left/right) |
| **EEG** | Continuous recordings during task (pre/post stimulation) |
| **Simulations** | Electric field (E-field) distribution from SimNIBS |
| **fMRI** | Structural & functional localization of right FEF |

---

## 🔍 Analysis Pipeline

### 1. Behavioural Analysis
- Fit **psychometric choice curves** to determine:
  - **PSE (Point of Subjective Equality)**
  - **Slope (sensitivity)**
- Compare **pre- vs post-tDCS** differences by polarity and participant.
- Evaluate **reaction time distributions** (mean and variance) .

### 2. EEG Analysis
- Preprocessing with `MNE-Python`:
  - Band-pass filtering (1–40 Hz)
  - Current Source Density Estimation via Surface Laplacian
- Compute **time–frequency decompositions** for alpha (8–12 Hz) and beta (13–30 Hz) bands.
- Contrast **pre/post tDCS** epochs.

### 3. Causal Inference (Difference-in-Differences Permutation)
- Implement **Difference-in-Differences (DiD)** testing:
  \[
  Δ_{tDCS} = (Post_{Anodal} - Pre_{Anodal}) - (Post_{Cathodal} - Pre_{Cathodal})
  \]
- Use **nonparametric permutation testing** (10,000 shuffles) to evaluate significance.
- Apply analysis at three scales:
  1. **Group level**
  2. **Individual level**
  3. **Session level (intra-individual variability)**

### 4. Electric Field Analysis
- Simulate individual E-field maps using *SimNIBS 4.0*.
- Extract regional mean intensity over the rFEF and surrounding areas.
---

## 📈 Key Findings

* **Group-level:** tDCS polarity significantly modulated reaction times and corresponding EEG responses, but not choice direction.
* **Inter-individual:** Effects were heterogeneous and often contradictory between participants.
* **Intra-individual:** Even within the same participant, session-level effects varied markedly, suggesting state-dependent or non-stationary influences.

These findings indicate that **tDCS effects are not reliably consistent across sessions or individuals**, challenging assumptions of generalizability in population-level analyses.

---

## 📚 Citation

If you use this repository or its methods, please cite:

```
@article{caie2025tDCS,
  title={Variability in the effects of transcranial direct current stimulation on free choice behaviour},
  author={Caie, Brandon and Blohm, Gunnar},
  year={2024},
  institution={Queen’s University, Centre for Neuroscience Studies}
  journal={bioRxiv}
  doi={https://doi.org/10.1101/2024.08.23.609379}

}
```

---

## 🧩 Reproducibility

All analysis pipelines are designed to be **fully reproducible** using provided data and scripts. The permutation testing results are included in the Zotero folder, as they required significant computational time. However, the scripts to perform the permutation testing are available in 'permutationTestingEEG.py' and 'permutationTestingPsychometrics.py', which run on the raw data included in the Zotero folder

---

## 🙏 Acknowledgments

This work was supported by the **Centre for Neuroscience Studies, Queen’s University**, **The Natural Sciences and Engineering Research Council of Canada (NSERC)**, and collaboration between **NSERC’s Collaborative Research and Training Experience (CREATE)** and **DFG’s International Research Training Groups (IRTG)**
We thank all participants and collaborators who contributed to data collection and analysis.

---

*For questions or contributions, please contact:*
📧 [brandon.caie@queensu.ca](mailto:brandon.caie@queensu.ca)

```
All data and analysis outputs are publicly available on Zenodo:

📦 Dataset DOI: https://zenodo.org/uploads/17486711

Contents

-Behavioural: Raw .csv files for behavioural data. Each file is named SS_PO_BL_DATE.csv, where SS is the subject identifier, PO is the polarity identifier (AN or CA), BL is the Block identifier (ex PR 1 for Pre Block 1, PO for Post Block 1)

-EEG: Preprocessed, trial-aligned EEG Files

- Psychometric Permutations: Output of the permutation testing for psychometrics.
  - Included are the 'true' difference-in-difference calculations based on the actual outcome of the experiment, and the 'null' distribution based on the permutation testing we calculated.
  - Each file is organized according to level (group, subject, session), data type (none referring to all data, Prev_Choice indicating splits based on the previous choice direciton, Prev_RT split by previous median RT, and Rep split between repetitions and alternations).

- EEG Permutations: Output of the EEG testing for psychometrics.

Data are shared under a CC-BY 4.0 license for academic and non-commercial use.```








