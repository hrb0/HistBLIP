# HistBLIP: A Vision-Language Model for Structured Pathology Report Generation in Clinical Settings

HistBLIP is a HistGen and BLIP-based model that analyzes pathology images and generates standardized reports for bladder cancer diagnosis.

## Preprocessing and Feature Extraction

### WSI Preprocessing
In this work, we adpoted and further accelerated [CLAM](https://github.com/mahmoodlab/CLAM) for preprocessing and feature extraction. We uploaded the minimal viable version of CLAM to this repo. For installation guide, we recommend to follow the original instructions [here](https://github.com/mahmoodlab/CLAM/blob/master/docs/INSTALLATION.md). To conduct preprocessing, please run the following commands:
```
cd HistBLIP
cd CLAM
conda activate clam
sh patching_scripts/tcga-wsi-report.sh
```

### Feature Extraction
To extract features of WSIs, please run the following commands:
```
cd HistGen
cd CLAM
conda activate clam
sh extract_scripts/tcga-wsi-report.sh
```

## HistBLIP WSI Report Generation Model

### Training
To try our model for training, validation, and testing, simply run the following commands:
```
cd HistGen
conda activate histgen
sh train_wsi_report_BLIP.sh
```

### Inference
To generate reports for WSIs in test set, you can run the following commands:
```
cd HistGen
conda activate histgen
sh test_wsi_report_BLIP.sh
```