# Detecting Patterns in Rhetoric

## Overview

This program is a combined bullshit and AI detector, designed to score pieces of text against both of these metrics

## Installation, setup and training

- Clone this repository
- Install python dependencies from `requirements.txt`
- Add training/testing data.
  - This should be stored in the following files:
    - `data_cleansing/merged/bullshit_training_all.csv`
    - `data_cleansing/merged/bullshit_testing_all.csv`
    - `data_cleansing/merged/ai_training_all.csv`
    - `data_cleansing/merged/ai_testing_all.csv`
    - This is the training and testing data for the two detectors. The CSV schema is `text,label0,label1`. Label 0 is the AI feature, Label 1 is the Bullshit feature. `1` is positive (AI/BS), `0` is negative (not AI/BS)
  - The code files `data_cleansing/clean.ipynb` and `data_cleansing/merge.ipynb` are for formatting the training data used for this project. For copyright reasons this data is not in this repository. If you have the appropriate data, these scripts will populate the required csv files, and expect the raw data in the following folders:
    - `human_bullshit/fake_news_files`
    - `human_bullshit/mission_files`
    - `human_bullshit/political_manifestos`
    - `human_bullshit/spam_msgs/spam.csv`
    - `human_bullshit/speech_files`
    - `human_bullshit/vision_mission_files`
    - `human_notshit/arxiv/arxiv_data_210930-054931.csv`
    - `human_notshit/arxiv/arxiv_data.csv`
    - `human_notshit/non-BS-BNC`
    - `human_notshit/wikipedia/processed`
    - `human_notshit/nature_articles/processed`
    - `human_bullshit/onion_news/processed`
    - `ai_detector_data/gemini_results_bs.zip`
    - `ai_detector_data/gemini_results_nbs.zip`
    - `ai_detector_data/gpt_articles.zip`
  - the wikipedia data can be gathered using `wikipedia/fetching_code/wikipedia_only.mjs`. The data should then be moved to the correct location. See `wikipedia/fetching_code/README.md` for more information
- To train all the active models, run `python3 src/train_all.py`
- The training scripts automatically save the trained artifacts to:
  - `trained_models/roberta_bs_model_v2/`
  - `trained_models/roberta_ai_model_v2/`
  - `trained_models/xgb_bs_model/`
  - `trained_models/xgb_ai_model/`

## Usage

To run the server:

```bash
python3 -m src.app
```
This hosts the web server on `localhost:5000`, which can then be accessed through the browser
