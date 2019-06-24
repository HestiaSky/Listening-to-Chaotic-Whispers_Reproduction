import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import jieba
import numpy as np

class D2V(object):

    def __init__(self, dirname, model_path):

        self.dirname = dirname
        self.model_path = model_path
        self.tagged_data = []

    def prepare_data(self):

        i = 0
        for news in os.listdir(self.dirname):
            f_path = os.path.join(self.dirname, news)
            data = open(f_path, 'rb').read()
            tag = news[:-4]
            words_list = jieba.lcut(data, cut_all=False)
            one_data = TaggedDocument(words=words_list, tags=[tag])
            self.tagged_data.append(one_data)
            i += 1
            if i % 100 == 0:
                print(i)

        print('Prepare done!')

    def train_data(self, max_epoch=15, vec_size=500, alpha=0.025):

        model = Doc2Vec(vector_size=vec_size, alpha=alpha, min_alpha=0.025, min_count=5,
                        dm=1, workers=30)

        model.build_vocab(self.tagged_data)

        for epoch in range(max_epoch):
            print('iteration {0}\n'.format(epoch))
            model.train(self.tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.epochs)
            # decrease learning rate
            model.alpha -= 0.0002
            # and reinitialize it
            model.min_alpha = model.alpha

        model.save(self.model_path)
        print('Model saved!')

if __name__ == '__main__':

    model = D2V('/home/lixinhang/bind_news_folder', '/home/lixinhang/d2v_model')
    model.prepare_data()
    model.train_data()
