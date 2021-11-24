from konlpy.tag import Mecab, Hannanum

#f = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/ko_mini_corpus"), "r", encoding="utf-8")
f = open('./corpus.txt', 'r')
sentence_list = f.readlines()
sentence_list = [pair.strip().split('||') for pair in sentence_list]

mecab = Mecab()
hannanum = Hannanum()

print(hannanum.pos('말했다'))
# for i in range(10):
#     print(sentence_list[i][0])
#     print(hannanum.pos(sentence_list[i][0]))
#print(mecab.pos('재고 수량 체크 하나 부탁할게요 유리아쥬 립밤으로'))
