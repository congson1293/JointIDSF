from typing import Text
import os

import json
import re


def mkdir(dir_name: Text):
    if not os.path.exists(dir_name):
        try:
            os.mkdir(dir_name)
        except Exception as e:
            print(e)


def seq2slot(input, ner):
    token2slot = dict()
    for i in ner:
        st, en = i['start'], i['end']
        x = input[st:en]
        x = x.split()
        token2slot[x[0]] = 'B_' + i['entity']
        if len(x) > 0:
            for j in x[1:]:
                token2slot[j] = 'I_' + i['entity']

    input_token = input.split()
    for i in input_token:
        if i not in token2slot:
            token2slot[i] = 'O'

    seqout = []
    for i in input_token:
        seqout.append(token2slot[i])
    return ' '.join(seqout)


def convertString2dict(string):
    subStr = re.findall(r'{(.+?)}', string)
    ner = []
    for j in subStr:
        j = '{' + j + '}'
        j = j.replace("'", '"')
        res = json.loads(j)
        ner.append(res)
    return ner


def save_data_to_file(intents, slots, labels, seqins, seqouts):
    mkdir('VPS_data')
    mkdir('VPS_data/word-level')
    mkdir('VPS_data/word-level/train')

    with open('VPS_data/word-level/intent_label.txt', 'w') as f:
        for i in intents:
            f.write(i)
            f.write('\n')
    f.close()

    with open('VPS_data/word-level/slot_label.txt', 'w') as f:
        for i in slots:
            f.write(i)
            f.write('\n')
    f.close()

    f = open('VPS_data/word-level/train/label', 'w')
    for i in labels:
        f.write(i)
        f.write('\n')
    f.close()

    f = open('VPS_data/word-level/train/seq.in', 'w')
    for i in seqins:
        f.write(i)
        f.write('\n')
    f.close()

    f = open('VPS_data/word-level/train/seq.out', 'w')
    for i in seqouts:
        f.write(i)
        f.write('\n')
    f.close()


def process(data):
    labels, seqins, seqouts = [], [], []
    for i in data:
        i = i.split(' - ')
        labels.append(i[0])
        seqins.append(i[1])
        if len(i) == 2:
            seqout = ' '.join(['O'] * len(i[1].split()))
            seqouts.append(seqout)
        else:
            ner = convertString2dict(i[2])
            seqout = seq2slot(i[1], ner)
            seqouts.append(seqout)

    intents = list(set(labels))
    slots = []
    for i in seqouts:
        i = i.split()
        slots += i
    slots = list(set(slots))

    save_data_to_file(intents, slots, labels, seqins, seqouts)


if __name__ == '__main__':
    import json

    with open('data.json', 'r') as fp:
        data = json.load(fp)

    process(data)
