import pickle
import bcolz
import numpy as np
import argparse
import os

def generate_glove_embeddings():
    words = []
    idx = 0
    word2idx = {}
    file_name = ''.join(glove_path.split('.')[:len(glove_path.split('.'))-1])
    vectors = bcolz.carray(np.zeros(1), rootdir=file_name+'.dat', mode='w')
    c=0
    with open(glove_path, 'rb') as f:
        for l in f:
            try:
                line = l.decode().split()
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx += 1
                vect = np.array(line[1:]).astype(np.float64)
                vectors.append(vect)
            except:
                c+=1
                continue

    vectors = bcolz.carray(vectors[1:].reshape((vocab_size, embedding_dimensions)), rootdir=file_name + '.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(file_name + '_words.pkl', 'wb'))
    pickle.dump(word2idx, open(file_name + '_idx.pkl', 'wb'))

def pickle_glove_embeddings():
    file_name = ''.join(glove_path.split('.')[:len(glove_path.split('.'))-1])
    vectors = bcolz.open(os.path.join(data_dir, file_name + '.dat'))[:]
    words = pickle.load(open(os.path.join(data_dir, file_name + '_words.pkl'), 'rb'))
    word2idx = pickle.load(open(os.path.join(data_dir, file_name + '_idx.pkl'), 'rb'))
    glove = {w: vectors[word2idx[w]] for w in words}
    target_vocab = list(pickle.load(open(os.path.join(data_dir, 'questions_vocab.pkl'), 'rb'))["word2idx"].keys())

    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, embedding_dimensions))

    for i, word in enumerate(target_vocab):
        try:
            weights_matrix[i] = glove[word]
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dimensions, ))

    with open(os.path.join(data_dir, f"word_embeddings_glove.pkl"), 'wb') as f:
            pickle.dump(weights_matrix, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VQA')
    parser.add_argument('--data_dir',               type=str,       help='directory of the preprocesses data', default='/scratch/an3729/datasets/')
    parser.add_argument('--glove_path',             type=str,       help='path to glove embeddings text file', default='glove.6B.300d.txt')
    parser.add_argument('--vocab_size',             type=int,       help='vocab size', default=400000)
    parser.add_argument('--embedding_dimensions',   type=int,       help='vocab size', default=300)
    parser.add_argument('--output_pickle_file_path',type=str,       help='path to save glove embeddings pickle file', default='word_embeddings_glove.pkl')

    args = parser.parse_args()
    data_dir = args.data_dir
    glove_path = args.glove_path
    output_path = args.output_pickle_file_path
    vocab_size = args.vocab_size
    embedding_dimensions = args.embedding_dimensions
    generate_glove_embeddings()
    pickle_glove_embeddings()