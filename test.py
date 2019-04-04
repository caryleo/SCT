import nltk

tokens = nltk.word_tokenize("The author of SCST helped me a lot when I tried to replicate the result. Great thanks. The att2in2 model can achieve more than 1.20 Cider score on Karpathy's test split (with self-critical training, bottom-up feature, large rnn hidden size, without ensemble)")
pos = nltk.pos_tag(tokens)
print(pos)
for p in pos:
    if p[1].startswith('N'):
        print(p[0], p[1])
