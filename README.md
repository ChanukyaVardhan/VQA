# VQA

## Repo Structure
    .
    ├── dataset.py            # dataset for the preprocessed data
    ├── download_datasets.sh  # script to download VQA2.0 data and extract them
    ├── main.py               # entry point for training, takes various arguments
    ├── models
    │	 └── baseline.py      # baseline model from VQA paper
    ├── preprocess.py         # preprocess the VQA2.0 data and save vocabulary files
    ├── README.md             # readme file
    ├── train.py              # function to train the model
    └── utils.py              # utility functions

## Download Datasets
Run `download_datasets.sh` script to download the VQA2.0 dataset from the official site. This includes the annotations, questions and the images. Set the appropriate dataset directory to download them to in the script.

## Preprocess Data
Run `python preprocess.py --data_dir ...` to preprocess the downloaded dataset. This script processes all the questions, annotations and saves each question example as a row in `image_id`\t`question`\t`answer` format in the processed txt files. This also saves the vocabulary of words in training questions mapping word to index and also index to the word, and also the frequencies of answers that will be used to construct the vocabulary for answers later.

