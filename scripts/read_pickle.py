import pickle
import dill

vocab_path = 'final-models/seq2seq/sri/py150/normal/lstm/checkpoints/Best_F1/input_vocab.pt'
with open(vocab_path, 'rb') as f:
    vocab = dill.load(f)
print(len(vocab.stoi))
print(len(vocab))


# file_path = 'datasets/augmented/random/tokens/sri/py150/train.pkl'
# # file_path = 'datasets/augmented/v2-3-z_rand_1-pgd_3_no-transforms.Replace-py150/tokens/sri/py150/test.pkl'

# with open(file_path, 'rb') as f:
#     data = pickle.load(f)

# print(len(data))
# print(len(data[0]))
# print(data[0][0] == data[0][1])
# print(data[0][0])
# print(data[0][1])
