import os
from tqdm import tqdm
from string import punctuation

SAVE_PATH = '../data/converted'
try:
    os.mkdir(SAVE_PATH)
except FileExistsError:
    print("data/converted folder already exists. Continue?(y/n)")
    cont = input()
    if cont == "n" or cont == "N":
        exit(0)


def read_file(path, name):
    with open(os.path.join(path, name), 'r') as f:
        review_list = f.readlines()

    return review_list


def remove_newline(text):
    if '\n' in text:
        return text.replace('\n', '')
    else:
        return text


def remove_punc(text):
    for c in punctuation:
        if c in text:
            text = text.replace(c, '')

    return text


def remove_sssss(text):
    if '<sssss> ' in text:
        return text.replace('<sssss> ', '')
    else:
        return text


def convert_file(path, name):
    review_list = read_file(path, name)

    for i, review in enumerate(tqdm(review_list)):
        content_list = review.split("\t")
        new_list = []
        for content in content_list:
            if len(content) > 0:
                new_list.append(remove_sssss(remove_newline(content)))

        text = new_list[3]
        label = int(new_list[2])

        with open(os.path.join(SAVE_PATH, name), 'a+') as f:
            f.write(text + '\t' + str(label) + '\n')


if __name__ == '__main__':
    convert_file('../data', 'train.txt')
    convert_file('../data', 'test.txt')
    convert_file('../data', 'dev.txt')
