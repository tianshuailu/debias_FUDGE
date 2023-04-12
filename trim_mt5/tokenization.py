"""
tokenize the dataset of 2 millions English and 2 millions Italian sentences for trimming the mt5-small model.
"""

from transformers import MT5Tokenizer

tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')
data = []
with open("en_and_it_4m.txt", "r") as my_file:
    for line in my_file:
        data.append(line)

f = open("en_and_it_4m.bpe.txt", "w")
for line in data[:-1]:
    output = tokenizer.tokenize(line)
    for token in output[:-1]:
        f.write(token + " ")
    f.write(output[-1] + '\n')

output = tokenizer.tokenize(data[-1])
for token in output[:-1]:
    f.write(token + " ")
f.write(output[-1])
f.close()
