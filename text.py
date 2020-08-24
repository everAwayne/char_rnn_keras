import re
import numpy as np
from keras.utils import to_categorical

class TextTokenizer:

    def __init__(self, file, max_vocab=3000, min_counting=None):
        if isinstance(file, str):
            with open(file, encoding='utf-8') as fd:
                text = fd.read()
        else:
            text = file.read()
        word_ls = re.split("", text)[1:-1]
        vocab_count = {}
        for word in word_ls:
            if word not in vocab_count:
                vocab_count[word] = 1
            else:
                vocab_count[word] += 1
        if min_counting:
            self.vocab_ls = [vocab for vocab, count in vocab_count.items() if count >= min_count]
        elif max_vocab:
            self.vocab_ls = [vocab for vocab, count in sorted(vocab_count.items(), key=lambda x:x[1], reverse=True)][:max_vocab]
        else:
            raise ValueError
        #self.vocab_ls.sort()
        self.vocab_to_int_map = {item:i for i, item in enumerate(self.vocab_ls)}
        self.int_to_vocab_map = {i:item for i, item in enumerate(self.vocab_ls)}
        self.text_array = np.array(self.encode(word_ls))

    @property
    def vocab_size(self):
        return len(self.vocab_ls) + 1

    def encode(self, word_ls):
        return [self.vocab_to_int_map.get(item, len(self.vocab_ls)) for item in word_ls]

    def decode(self, ls):
        return [self.int_to_vocab_map.get(item, "None") for item in ls]

    def training_data(self, time_step):
        size = len(self.text_array) // time_step
        x = self.text_array[: size * time_step]
        y = np.roll(x, -1)
        #x.reshape(-1, time_step)
        #y.reshape(-1, time_step)
        #return x.reshape((-1, time_step)), y.reshape((-1, time_step))[:, time_step-1]
        #return x.reshape((-1, time_step)), to_categorical(y.reshape((-1, time_step))[:, time_step-1], self.vocab_size)
        return x.reshape((-1, time_step)), to_categorical(y.reshape((-1, time_step)), self.vocab_size)

    def training_data_generator(self, num_seqs, time_steps):
        batch_size = num_seqs * time_steps
        num_batches = len(self.text_array) // batch_size
        x = self.text_array[: batch_size * num_batches]
        y = np.roll(x, -1)
        x = x.reshape((num_seqs, 1, -1))
        y = y.reshape((num_seqs, 1, -1))
        arr = np.concatenate((x,y), 1)
        while True:
            np.random.shuffle(arr)
            for n in range(0, arr.shape[-1], time_steps):
                x = arr[:, 0, n:n + time_steps]
                y = np.zeros_like(x)
                y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
                yield x, to_categorical(y, self.vocab_size)
                #yield arr[:, 0, n:n + time_steps], to_categorical(arr[:, 1, n:n + time_steps], self.vocab_size)
                #yield arr[:, 0, n:n + time_steps], arr[:, 1, n:n + time_steps]
