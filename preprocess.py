import argparse
import collections
import json
import numpy as np
import os
import pickle
import string

random_seed = 43
np.random.seed(random_seed)

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

def save_preprocessed_data(data_dir, output_file, image_data, qqa):
    with open(os.path.join(data_dir, output_file), 'w') as out:
        for image_id, questions in image_data.items():
            for question in questions:
                question_id = question['question_id']
                ques        = preprocess_text(qqa[question_id]['question'])
                answer      = question['multiple_choice_answer']
                out.write(str(image_id) + "\t" + ques + "\t" + answer + "\n")

def preprocess(data_dir, num_train_samples = 20000, num_val_samples = 10000, num_test_samples = 20000):
    print("Preprocessing VQA 2.0 dataset!")

    ques_file         = "v2_OpenEnded_mscoco_train2014_questions.json"
    annot_file        = "v2_mscoco_train2014_annotations.json"
    train_output_file = "train_data.txt"
    val_output_file   = "val_data.txt"
    test_output_file  = "test_data.txt"

    question_file     = os.path.join(data_dir, 'questions', ques_file)
    annotation_file   = os.path.join(data_dir, 'annotations', annot_file)

    annotations       = json.load(open(annotation_file, 'r'))
    questions         = json.load(open(question_file, 'r'))
    
    imgToQA           = {ann['image_id']: [] for ann in annotations['annotations']}
    qqa               = {ann['question_id']: [] for ann in annotations['annotations']}
    for ann in annotations['annotations']:
        imgToQA[ann['image_id']] += [ann]
    for ques in questions['questions']:
        qqa[ques['question_id']]  = ques
    
    # numQ = [len(x) for k, x in imgToQA.items()]
    # min(numQ) # -> 3

    num_samples       = num_train_samples + num_val_samples + num_test_samples
    num_ques_image    = 3

    if num_samples > len(imgToQA):
        print(f"Requested sampling of {num_samples} images which is more than the actual number of {len(imgToQa)} images!")
    
    print(f"Sampling {num_samples} images from train2014 folder!")
    imgToQAKeys       = imgToQA.keys()
    sampled_images    = np.random.choice(list(imgToQAKeys), num_samples, replace = False)
    train_images, val_images, test_images, _ = \
        np.split(sampled_images,[num_train_samples, num_train_samples + num_val_samples, num_samples])

    print(f"Sampling {num_train_samples} training images!")
    train_images      = {k: imgToQA[k] for k in train_images}
    print(f"Sampling {num_val_samples} validations images!")
    val_images        = {k: imgToQA[k] for k in val_images}
    print(f"Sampling {num_test_samples} testing images!")
    test_images       = {k: imgToQA[k] for k in test_images}

    print(f"Sampling {num_ques_image} questions per image!")
    train_images      = {k: np.random.choice(d, num_ques_image, replace = False) for k, d in train_images.items()}
    val_images        = {k: np.random.choice(d, num_ques_image, replace = False) for k, d in val_images.items()}
    test_images       = {k: np.random.choice(d, num_ques_image, replace = False) for k, d in test_images.items()}
    
    save_preprocessed_data(data_dir, train_output_file, train_images, qqa)
    save_preprocessed_data(data_dir, val_output_file, val_images, qqa)
    save_preprocessed_data(data_dir, test_output_file, test_images, qqa)
    
    print("Completed preprocessing VQA 2.0 dataset!")

def save_answer_freqs():
    with open(os.path.join(data_dir, 'train_data.txt'), 'r') as f:
        data = f.read().strip().split('\n')

    answers              = [x.split('\t')[2].strip() for x in data]
    answer_freq          = dict(collections.Counter(answers))
    print(f"Total number of answers in training data - {len(answer_freq)}!")
    
    with open(os.path.join(data_dir, 'answers_freqs.pkl'), 'wb') as f:
        pickle.dump(answer_freq, f)

    print("Saving answer frequencies from train data!")

def save_vocab_questions(min_word_count = 3):
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

data_dir         = "/scratch/crg9968/datasets/"

preprocess(data_dir)
save_answer_freqs()
save_vocab_questions()
