from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dropout, Dense, LSTM
from keras.initializers import TruncatedNormal
from keras import optimizers
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np

class CharRnn:

    def __init__(self, num_classes, num_seqs, time_steps, use_embedding, embedding_size, lstm_units,
                 num_layers, learning_rate, dropout_rate, save_file=None):
        self.num_classes = num_classes
        inputs = Input(batch_shape=(num_seqs, time_steps))
        if use_embedding:
            x = Embedding(num_classes, embedding_size, name='embedding')(inputs)
        else:
            x = to_categorical(inputs, num_classes)
        layer_num = 1
        while layer_num <= num_layers:
            x = LSTM(lstm_units, return_sequences=True, dropout=dropout_rate,
                     stateful=True, name='lstm_'+str(layer_num))(x)
            x = Dropout(dropout_rate, name='dropout_'+str(layer_num))(x)
            layer_num += 1
        predictions = Dense(num_classes, activation='softmax', kernel_initializer=TruncatedNormal(stddev=0.1),
                            bias_initializer='zeros', name='softmax')(x)
        self.model = Model(inputs=inputs, outputs=predictions)
        '''
        self.model = Sequential()
        if use_embedding:
            self.model.add(Embedding(num_classes, embedding_size, name='embedding'))
        layer_num = 1
        while layer_num <= num_layers:
            if layer_num == 1:
                if not use_embedding:
                    self.model.add(LSTM(lstm_units, return_sequences=True, input_dim=num_classes, name='lstm_'+str(layer_num)))
                else:
                    self.model.add(LSTM(lstm_units, return_sequences=True, name='lstm_'+str(layer_num)))
            self.model.add(Dropout(dropout_rate, name='dropout_'+str(layer_num)))
            layer_num += 1
        self.model.add(Dense(num_classes, activation='softmax', name='softmax'))
        '''
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizers.Adam(lr=learning_rate, clipnorm=5),
                           metrics=['accuracy'])
        if save_file:
            #self.model = load_model(save_file)
            self.model.load_weights(save_file)
        print(self.model.summary())

    def train(self, train_g, steps_per_epoch, epochs, save_file=None):
        #self.model.fit_generator(train_g, steps_per_epoch=steps_per_epoch, epochs=epochs)
        i = 1
        for x, y in train_g:
            loss = self.model.train_on_batch(x, y)
            i += 1
            if i%10 == 0:
                print(i, loss)
            if i >=10000:
                break
        if save_file:
            self.model.save(save_file)
            #self.model.save_weights(save_file)

    def generate(self, start_array, text_len):
        result = [i for i in start_array]
        preds = np.ones((self.num_classes, ))
        for c in start_array:
            x = np.zeros((1, 1))
            x[0, 0] = c
            preds = self.model.predict(x)
            #print(preds.shape)

        c = self.pick_top_n(preds, preds.shape[-1])
        result.append(c)

        for i in range(text_len):
            x = np.zeros((1, 1))
            x[0, 0] = c
            preds = self.model.predict(x)
            c = self.pick_top_n(preds, preds.shape[-1])
            result.append(c)
        return result

    def pick_top_n(self, preds, vocab_size, top_n=5):
        p = np.squeeze(preds)
        # 将除了top_n个预测值的位置都置为0
        p[np.argsort(p)[:-top_n]] = 0
        # 归一化概率
        p = p / np.sum(p)
        # 随机选取一个字符
        #print("----", p.shape)
        #print("----", vocab_size)
        c = np.random.choice(vocab_size, 1, p=p)[0]
        return c
