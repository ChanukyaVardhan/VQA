"""
Given the directory where the questions & annotations files are saved, this generates a processed 
data file (.txt) in the `image_id`\t`question`\t`answer` format (question and answer are 
space separated strings). The files are stored in the same data directory passed as input

python preprocess.py --data_dir ../Dataset
"""

import argparse
import collections
import json
import os
import pickle
import string

def preprocess_text(text):
    """
        Converts a string to lower case, removes punctuations.
    """
    text_token_list = text.strip().split(',')
    text  = ' '.join(text_token_list)

    # Remove punctuations
    table = str.maketrans('', '', string.punctuation)
    words = text.strip().split()
    words = [w.translate(table) for w in words]

    # Set to lowercase & drop empty strings
    words = [word.lower() for word in words if word != '' and word != 's']
    
    text  = ' '.join(words)
    return text

def preprocess(data_dir, mode = 'train'):
    """
        Reads the question and annotation files in data_dir given a mode (train/val),
        and processes them in `image_id`\t`question`\t`answer` format, and saves them
        in .txt files in the same data_dir
    """
    print(f"Preprocessing VQA 2.0 dataset for {mode}!")

    ques_file         = f"v2_OpenEnded_mscoco_{mode}2014_questions.json"
    annot_file        = f"v2_mscoco_{mode}2014_annotations.json"
    output_file       = f"{mode}_data.txt"

    question_file     = os.path.join(data_dir, 'questions', ques_file)
    annotation_file   = os.path.join(data_dir, 'annotations', annot_file)

    annotations       = json.load(open(annotation_file, 'r'))
    questions         = json.load(open(question_file, 'r'))
    print(f"Total number of questions in {mode} set - {len(questions['questions'])}")

    with open(os.path.join(data_dir, output_file), 'w') as out:
        for question, annotation in zip(questions['questions'], annotations['annotations']):
            image_id  = question['image_id']
            ques      = preprocess_text(question['question'])
            answer    = annotation['multiple_choice_answer'] # Most frequent ground truth answer
            answers   = [a['answer'] for a in annotation['answers']]
            answers   = '^'.join(answers)
            out.write(str(image_id) + "\t" + ques + "\t" + answer + "\t" + answers + "\n")
    
    print("Completed preprocessing VQA 2.0 dataset!")

def save_answer_freqs(data_dir):
    """
        Reads the preprocessed train_data.txt file in data_dir, calculates the
        frequences of answers, and saves them in answers_freqs.pkl file
    """
    with open(os.path.join(data_dir, 'train_data.txt'), 'r') as f:
        data = f.read().strip().split('\n')

    answers              = [x.split('\t')[2].strip() for x in data]
    answer_freq          = dict(collections.Counter(answers))
    print(f"Total number of answers in training data - {len(answer_freq)}!")
    
    with open(os.path.join(data_dir, 'answers_freqs.pkl'), 'wb') as f:
        pickle.dump(answer_freq, f)

    print("Saving answer frequencies from train data!")

def save_vocab_questions(data_dir, min_word_count = 3):
    """
        Reads the preprocessed train_data.txt file in data_dir and saves the vocabulary
        of words for the questions in questions_vocab.pkl file
    """
    with open(os.path.join(data_dir, 'train_data.txt'), 'r') as f:
        data = f.read().strip().split('\n')

    questions           = [x.split('\t')[1].strip() for x in data]
    words               = [x.split() for x in questions]
    words               = [w for x in words for w in x]
    word_index          = [w for (w, freq) in collections.Counter(words).items() if freq > min_word_count]
    word_index          = {w: i+2 for i, w in enumerate(word_index)}
    word_index["<pad>"] = 0
    word_index["<unk>"] = 1
    index_word          = {i: x for x, i in word_index.items()}
    print(f"Total number of words in questions in training data - {len(word_index)}!")

    with open(os.path.join(data_dir, 'questions_vocab.pkl'), 'wb') as f:
        pickle.dump({"word2idx": word_index, "idx2word": index_word}, f)

    print("Saving vocab for words in questions!")

def main():
    parser = argparse.ArgumentParser(description='Preprocess VQA dataset')
    parser.add_argument('--data_dir', type=str, help='directory to preprocesses data', default='/scratch/crg9968/datasets')
    args = parser.parse_args()

    for mode in ['train', 'val']:
        preprocess(args.data_dir, mode)
    save_answer_freqs(args.data_dir)
    save_vocab_questions(args.data_dir)

if __name__ == '__main__':
    main()
