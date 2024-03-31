# Zindi 6th Position: Location Mention Recognition

## Overview
This repository contains the work and methodologies that led to a 6th-place finish in the Zindi Location Mention Recognition Competition. The project focuses on the application, evaluation, and enhancement of the Suweilah Named Entity Recognition (NER) model, particularly aimed at extracting and accurately identifying location mentions within textual data.

## Repository Structure
- `suweilah.py`: Implements the Suweilah NER model for entity extraction from textual data.
- `enforce_entity_suweilah.py`: Refines the Suweilah model outputs by correcting and enhancing detected entity mentions.
- `matches31.py`: Analyzes the accuracy of entity detection, focusing on identifying and quantifying false positives.
- `p12.py`: Main script that integrates data preprocessing, entity processing, and evaluation of the NER model's performance.
- `data/`: Contains essential datasets and generated analysis files:
  - `check_detection.csv`: Initial dataset for analysis.
  - `fp_analysis.csv`: Analysis output detailing false positives in entity detection.
  - `malformed_suweilah.csv`: Lists entities that were incorrectly detected, requiring corrections.
  - `suweilah_test.csv`: Test dataset used for model evaluation.
- `README.md`: Provides an overview and guide for this repository.

## Installation

To get started with this project, clone the repository to your local machine:

```bash
git clone https://github.com/muhammad-luay/Zindi-6th-Position-Location-Mention-Recognition-.git
cd Zindi-6th-Position-Location-Mention-Recognition-
```


## Contributing

We welcome contributions to improve the methodologies and models used in this project. If you have suggestions or improvements, please open an issue to discuss your ideas or directly submit a pull request.

## License

This project is open-sourced under the MIT License

## Acknowledgments

- Zindi for hosting the competition.

