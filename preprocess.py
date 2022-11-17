import argparse
import collections
import json
import os
import pickle
import string

def preprocess_text(text):
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
    print(f"Preprocessing VQA 2.0 data for {mode} dataset!")
    
    ques_file         = f"v2_OpenEnded_mscoco_{mode}2014_questions.json"
    annot_file        = f"v2_mscoco_{mode}2014_annotations.json"
    output_file       = f"{mode}_data.txt"

    question_file     = os.path.join(data_dir, 'questions', ques_file)
    annotation_file   = os.path.join(data_dir, 'annotations', annot_file)

    annotations       = json.load(open(annotation_file, 'r'))
    questions         = json.load(open(question_file, 'r'))

    with open(os.path.join(data_dir, output_file), 'w') as out:
        for question, annotation in zip(questions['questions'], annotations['annotations']):
            image_id  = question['image_id']
            # ANY MORE PREPROCESSING TO DO FOR QUESTIONS?
            ques      = preprocess_text(question['question'])
            # SHOULD WE PREPROCESS THE ANSWERS AS WELL?
            answer    = annotation['multiple_choice_answer']
            out.write(str(image_id) + "\t" + ques + "\t" + answer + "\n")

def save_answer_freqs():
    with open(os.path.join(data_dir, 'train_data.txt'), 'r') as f:
        data = f.read().strip().split('\n')
    
    answers              = [x.split('\t')[2].strip() for x in data]
    answer_freq          = dict(collections.Counter(answers))

    with open(os.path.join(data_dir, 'answers_freqs.pkl'), 'wb') as f:
        pickle.dump(answer_freq, f)

    print(f"Saving answer frequencies from train data!")

def save_vocab_questions(min_word_count = 5):
    with open(os.path.join(data_dir, 'train_data.txt'), 'r') as f:
        data = f.read().strip().split('\n')

    questions           = [x.split('\t')[1].strip() for x in data]
    words               = [x.split() for x in questions]
    max_length          = max([len(x) for x in words])
    words               = [w for x in words for w in x]
    c                   = collections.Counter(words)
    word_index          = {w: i+2 for i, (w, freq) in enumerate(c.items()) if freq > min_word_count}
    word_index["<pad>"] = 0
    word_index["<unk>"] = 1
    index_word          = {i: x for x, i in word_index.items()}
    
    with open(os.path.join(data_dir, 'questions_vocab.pkl'), 'wb') as f:
        pickle.dump({"word2idx": word_index, "idx2word": index_word, "max_length": max_length}, f)

    print(f"Saving vocab for words in questions!")

data_dir         = "/scratch/crg9968/datasets/"

preprocess(data_dir, 'train')
preprocess(data_dir, 'val')
save_answer_freqs()
save_vocab_questions()
