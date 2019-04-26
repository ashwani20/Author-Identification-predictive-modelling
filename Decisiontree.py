import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

def is_data_pure(data):
    label_column = data[:, -1]
    uniquelist = []
    for eachlabel in label_column:
        if eachlabel not in uniquelist:
            uniquelist.append(eachlabel)
    return True if len(uniquelist) == 1 else False


def classify(data):
    label_column = data[:, -1]
    label_dict = {}
    for eachlabel in label_column:
        if eachlabel not in label_dict:
            label_dict[eachlabel] = 1
        else:
            label_dict[eachlabel] = label_dict.get(eachlabel)+1
    best_class = ''
    count = 0
    for key, value in label_dict.items():
        if value > count or count == 0:
            best_class = key
            count = value
    return best_class


def generate_all_splits(data):
    split_dictionary = {}
    col_num = data.shape[1] - 1
    row_list = data[:, 0:(col_num - 1)]

    for i in range(col_num):
        split_dictionary[i] = []
        col_vallist = data[:, i:i + 1]
        for j in range(1, len(col_vallist)):
            avg_sum = round((sum(col_vallist[j] + col_vallist[j - 1])) / 2, 2)
            split_dictionary[i].append(avg_sum)
    return split_dictionary


def partition(data, split_column, split_value):
    split_column_values = data[:, split_column]
    return data[split_column_values <= split_value], data[split_column_values >  split_value]


def cal_entropy(data):
    label_column = data[:, -1]
    label_dict = {}
    for eachlabel in label_column:
        if eachlabel not in label_dict:
            label_dict[eachlabel] = 1
        else:
            label_dict[eachlabel] = label_dict.get(eachlabel) + 1
    total_count = sum(label_dict.values())
    entropy = 0
    for eachkey in label_dict.keys():
        temp = round(label_dict[eachkey] / total_count, 3)
        entropy += - temp * np.log(temp)
    return entropy


def cal_net_entropy(left_branch, right_branch):
    total = len(left_branch) + len(right_branch)
    prob_left_branch, p_right_branch = len(left_branch)/total, len(right_branch)/total
    overall_entropy = (prob_left_branch * cal_entropy(left_branch)
                       + p_right_branch * cal_entropy(right_branch))
    return overall_entropy


def find_best_split(data, potential_splits):
    overall_entropy = float("inf")
    columnlist = potential_splits.keys()
    for column_index in columnlist:
        for value in potential_splits[column_index]:
            left_branch, right_branch = partition(data, column_index, value)
            current_overall_entropy = cal_net_entropy(left_branch, right_branch)
            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value


def generate_decision_tree(data, column_list, min_samples=2, max_depth=4, level = 0):
    level += 1
    if (is_data_pure(data)) or (len(data) < min_samples) or (level == max_depth):
        classification = classify(data)
        return classification
    else:
        potential_splits = generate_all_splits(data)
        split_column, split_value = find_best_split(data, potential_splits)
        left_branch, right_branch = partition(data, split_column, split_value)

        feature = column_list[split_column]
        question = "{} <= {}".format(feature, split_value)
        sub_tree = {question: []}

        true_result = generate_decision_tree(left_branch, column_list, min_samples, max_depth, level)
        false_result = generate_decision_tree(right_branch, column_list, min_samples, max_depth, level)

        if true_result != false_result:
            sub_tree[question].append(true_result)
            sub_tree[question].append(false_result)
        else:
            sub_tree = true_result

        return sub_tree


def classify_example(example, tree):
    question = list(tree.keys())[0]
    splitlist = question.split(" ")
    feature = splitlist[0]
    comparison_operator = splitlist[1]
    value = splitlist[2]

    answer = tree[question][1] if example[feature] > float(value) else tree[question][0]

    return classify_example(example, answer) if isinstance(answer, dict) else answer


def calculate_accuracy(df, tree):
    df["classification"] = df.apply(classify_example, axis=1, args=(tree,))
    return (df.loc[:, "classification"] == df.loc[:, "author_name"]).mean()