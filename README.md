# Robust CIFAR-10 classification from 500 images (DLP Assignment 2, Section B)

A small Inception-style CNN (64,402 parameters, trained from scratch) is trained on **400 CIFAR-10 images** (100 more for validation) and tested on a **corrupted version of the 10,000-image test set** (rotation, brightness/contrast, blur, noise, ~25% occlusion and one hidden corruption: random RGB channel swap).

Everything (data split, corrupted test set, model, augmentation, ablation, error analysis, test-time augmentation, summary file) is in one notebook:
`DL_ASS01_XXF_YYYY.ipynb` (rename to your batch and roll number).

## How to reproduce
1. Open the notebook in **Google Colab** (Runtime → Change runtime type → GPU is recommended, but not required).
2. Run **Runtime → Run all**, from the first cell to the last, in order. Do not run cells separately.
3. CIFAR-10 is downloaded automatically with `torchvision` (and cached as `cifar10.npz`).
4. At the end the notebook writes into `results/`: `Results_Summary.docx`, `summary.json`, `ablation_table.csv`, `subset_accuracy.csv`, `hidden_corruption.csv`.
5. Finished runs are cached in `results/run_*.pkl`. To retrain everything from scratch, set `USE_CACHE = False` in the settings cell (or delete the `results/` folder).

All randomness is controlled: training seeds `0, 1, 2`, data split seed `42`, corruption seed `1234`. Maximum 60 epochs per run, batch size 50. Reported numbers are mean ± std over the 3 training seeds. Exact digits can differ slightly between CPU and GPU runtimes.

## Method in short
* **Model:** 5×5 conv stem → 3 Inception modules (1×1, 3×3, 5×5, pooling branches with 1×1 reductions) → global average pooling → dropout → linear. BatchNorm after every convolution.
* **Training:** SGD + Nesterov momentum, one-cycle learning rate, weight decay, early stopping on validation accuracy. Hyper-parameters chosen on the validation split only. The test sets are evaluated only after training.
* **Augmentation (on the fly, training data only):** rotation, scale, shift, horizontal flip, brightness/contrast/saturation jitter, random grayscale, channel shuffle, blur, noise, Cutout, CutMix, label smoothing.

## Results (accuracy %, mean ± std over 3 seeds)
| Configuration | Val | Clean test | Shifted test |
|---|---|---|---|
| No augmentation | 51.7 ± 1.2 | 41.5 ± 0.4 | 18.0 ± 0.5 |
| Geometric only | 50.7 ± 1.2 | 41.2 ± 0.4 | 17.9 ± 0.8 |
| Photometric only | 50.3 ± 1.9 | 35.9 ± 0.8 | 18.4 ± 0.8 |
| Geometric + photometric | 57.0 ± 0.0 | 43.1 ± 0.8 | 21.9 ± 0.9 |
| + Cutout | 57.3 ± 1.2 | 41.2 ± 1.7 | 22.6 ± 0.9 |
| + CutMix + label smoothing | 54.0 ± 1.4 | 41.9 ± 0.7 | 22.0 ± 0.6 |
| **Final model (tuned)** | 56.7 ± 0.5 | 42.7 ± 1.0 | **23.0 ± 0.4** |

Largest single improvement: combining geometric and photometric augmentation (neither helped alone). Test-time augmentation (8 views) gave no gain (22.9%) at about 8× the inference cost.

## Notes
* The official `shifted_test` file was not available, so the notebook builds its own corrupted test set following the assignment description. To use an official file, set `OFFICIAL_SHIFTED_PATH` in the first code cell.
* Author: NAME, ROLL NO.
