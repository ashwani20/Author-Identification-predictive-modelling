import pandas as pd
import glob
import math
import os
import pickle
from sys import argv
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from itertools import groupby
from random import shuffle
from logisticClassifier import *
from Decisiontree import *
from pprint import pprint
import os.path




average_word_Length = 0
average_sentence_length = 0
average_stopword_count = 0
type_token_ratio = 0
average_comma_count = 0
hapax_legomenon_ratio = 0
function_words = 0
author_name = ''
df_record_set = None
record_set = []
lr_model = None
tree = None


def words(item):
    return filter(lambda word: len(word) > 0, [word.strip("0123456789!:,.?(){}[]") for word in item.split()])

def yule(entry):
    word_dict = {}
    stemmer = PorterStemmer()
    for eachword in words(entry):
        eachword = stemmer.stem(eachword).lower()
        try:
            word_dict[eachword] += 1
        except KeyError:
            word_dict[eachword] = 1

    M1 = float(len(word_dict))
    M2 = sum([len(list(group)) * (freq ** 2) for freq, group in groupby(sorted(word_dict.values()))])

    try:
        return (M1 * M1) / (M2 - M1)
    except ZeroDivisionError:
        return 0


def cal_word_frequency(content):
    word_frequency = {}
    for word in content.split(' '):
        word = word.lower()
        if word not in word_frequency:
            word_frequency[word] = 1
        else:
            word_frequency[word] = 1 + word_frequency.get(word)
    return word_frequency


def generate_parameters(novel_content, author):
    global average_word_Length, average_sentence_length, \
        average_stopword_count, author_name, average_comma_count
    author_name = author
    stop_words = set(stopwords.words("english"))
    lineNum = 0
    comma_count = 0
    counter = len(novel_content)//10
    stopword_count = 0
    yule_index = 0
    authorId = 1 if author == 'Arthur' else 0
    i = 0
    while i<10:
        content = novel_content[lineNum:lineNum+counter]
        sentencelist = sent_tokenize(content)
        wordlist = word_tokenize(content)
        word_len, sentence_len, unique_count = 0, 0, 0
        for sentence in sentencelist:
            sentence_len += len(sentence)
        average_sentence_length = round(sentence_len/len(sentencelist),2)
        for word in wordlist:
            word_len += len(word)
            if word in stop_words:
                stopword_count += 1
            if word == ',':
                comma_count += 1
        average_word_Length = round(word_len/len(wordlist),2)
        average_stopword_count = round(stopword_count/len(wordlist),2)
        average_comma_count = round(comma_count / len(wordlist), 2)
        yule_index = round(yule(content),2)
        word_frequency = cal_word_frequency(content)
        for k, v in word_frequency.items():
            if v == 1:
                unique_count += 1
        type_token_ratio = round(unique_count/len(wordlist),2)
        record_set.append([average_word_Length, average_sentence_length, average_stopword_count,
                           yule_index, average_comma_count, authorId])
        lineNum += counter + 1
        i += 1


def read_file(files, author):
    global average_word_Length, average_sentence_length, average_stopword_count, \
        type_token_ratio, hapax_legomenon_ratio, function_words, average_comma_count, author_name
    for name in files:
        with open(name) as file:
            print(file.readline())
            linenumber = 0
            novel_content = ''.join(map(str, file.readlines())).replace('\n',' ')
            generate_parameters(novel_content, author)


def main():
    global average_word_Length, average_sentence_length, type_token_ratio, \
        function_words, author_name, df_record_set, record_set, \
        average_comma_count

    model_choice = 0

    train_flag = False
    predict_flag = False
    filename = ''
    if '-train' in argv:
        train_flag = True
    elif '-predict' in argv:
        predict_flag = True
        for eacharg in argv:
            if '.txt' in str(eacharg):
                filename = eacharg
        if len(filename) == 0:
            filename = input('please enter file name - ')
    else:
        print("Incorrect format")
    if train_flag:
        # path_Arthur = 'C:/Users/User/PycharmProjects/FISLab2/venv/Arthur Conan Doyle/*.txt'
        # path_Herman = 'C:/Users/User/PycharmProjects/FISLab2/venv/Herman Melville/*.txt'
        path_Arthur = 'Arthur Conan Doyle/*.txt'
        path_Herman = 'Herman Melville/*.txt'
        files = glob.glob(path_Arthur)
        read_file(files, 'Arthur')

        files = glob.glob(path_Herman)
        read_file(files, 'Herman')


        shuffle(record_set)
        bound = math.floor(len(record_set)*0.8)
        train_data = record_set[0:bound]
        test_data = record_set[bound + 1: len(record_set)-1]

        df_record_set = pd.DataFrame(record_set, columns=['average_word_Length',
                                                          'average_sentence_length',
                                                          'yule_index',
                                                          'function_words',
                                                          'average_comma_count',
                                                          'author_name'])
        try:
            os.remove('author_identification.csv')
        except OSError:
            pass

        df_record_set.to_csv('author_identification.csv')

        model_choice = int(input("Press 1 for building decicion tree\nPress 2 for building "
                       "logistic classifer\nPress 3 for both\n"))
        if model_choice == 1:
            build_decision_tree()
        elif model_choice == 2:
            build_logistic_model()
        elif model_choice ==3:
            build_decision_tree()
            build_logistic_model()
        else:
            print("incorrect model choice")

        try:
            os.remove('choice.txt')
        except OSError:
            pass

        # with open('choice.txt','w') as f:
        #     f.write(model_choice)

    elif predict_flag:
        df_test = pd.read_csv("test.csv")
        df_test.drop(columns=["Unnamed: 0"], axis=1, inplace=True)

        if os.path.isfile('lr_model.pickle'):
            with open('lr_model.pickle', 'rb') as handle:
                lr_model = pickle.load(handle)
            # print(df.head())
            df_test_X = df_test.iloc[:, 0:4]
            df_test_Y = df_test.iloc[:, 5]
            preds = lr_model.predict(df_test_X, 0.5)
            print('Accuracy of Logistic Classifier - ', (preds == df_test_Y).mean())

        if os.path.isfile('decision_tree.pickle'):
            with open('decision_tree.pickle', 'rb') as handle:
                decision_tree = pickle.load(handle)

            pprint(decision_tree)
            print(type(decision_tree))
            accuracy = calculate_accuracy(df_test, decision_tree)
            print('Accuracy of decision tree - ', accuracy)



    else:
        print("Incorrect format for running the program")


def build_decision_tree():
    global tree
    df = pd.read_csv("author_identification.csv")
    df.drop(columns=["Unnamed: 0"], axis=1, inplace=True)

    bound = math.floor(df.shape[0] * 0.8)
    train_data = df.iloc[0:bound, ]
    test_data = df.iloc[bound:df.shape[0], ]

    column_list = train_data.columns
    tree = generate_decision_tree(train_data.values, column_list, max_depth=4)
    pprint(tree)
    example = test_data.iloc[0]

    with open('decision_tree.pickle', 'wb') as handle:
        pickle.dump(tree, handle, protocol=pickle.HIGHEST_PROTOCOL)



    accuracy = calculate_accuracy(test_data , tree)
    print('Accuracy of decision tree - ', accuracy)


def build_logistic_model():
    global lr_model
    df = pd.read_csv("author_identification.csv")
    df.drop(columns=["Unnamed: 0"], axis=1, inplace=True)
    # print(df.head())
    X = df.iloc[:, 0:4]
    Y = df.iloc[:, 5]

    bound = math.floor(X.shape[0] * 0.8)

    X_train = X.iloc[0:bound, ]
    Y_train = Y.iloc[0:bound, ]

    X_test = X.iloc[bound:X.shape[0], ]
    Y_test = Y.iloc[bound:Y.shape[0], ]
    lr_model = LogisticRegression(0.1, 300000)

    lr_model.fit(X_train, Y_train)

    with open('lr_model.pickle', 'wb') as handle:
        pickle.dump(lr_model, handle, protocol=pickle.HIGHEST_PROTOCOL)


    preds = lr_model.predict(X_test, 0.5)


    print('Accuracy of Logistic Classifier - ', (preds == Y_test).mean())


if __name__ == '__main__':
    # build_logistic_model()
    # build_decision_tree()
    main()