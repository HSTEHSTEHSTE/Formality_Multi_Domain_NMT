from Korpora import Korpora

corpus = Korpora.load('korean_parallel_koen_news')
print(corpus)
with open('corpus.txt', 'w') as f:
    for i in range(94123):
        f.write(corpus.train[i].text + '||' + corpus.train[i].pair + '\n')
    for i in range(1000):
        f.write(corpus.dev[i].text + '||' + corpus.dev[i].pair + '\n')
    for i in range(2000):
        f.write(corpus.test[i].text + '||' + corpus.test[i].pair + '\n')
    