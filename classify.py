import os
import math

# These first two functions require os operations and so are completed for you
# Completed for you
from ctypes import sizeof


def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d.startswith('.'):
            # ignore hidden files
            continue
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d + "/"
        files = os.listdir(directory + subdir)
        for f in files:
            if f.startswith('.'):
                continue
            bow = create_bow(vocab, directory + subdir + f)
            dataset.append({'label': label, 'bow': bow})
    return dataset


# Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        if d.startswith('.'):
            # ignore hidden files
            continue
        subdir = d if d[-1] == '/' else d + '/'
        files = os.listdir(directory + subdir)
        for f in files:
            with open(directory + subdir + f, 'r') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])


# The rest of the functions need modifications ------------------------------
# Needs modifications
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {"None": 0}
    with open(filepath) as doc:
        for word in doc:
            word = word.strip()
            if word in vocab and len(word) > 0:
                if word in bow:
                    bow[word] += 1
                else:
                    bow[word] = 1
            else:
                bow["None"] += 1
    return bow


# Needs modifications
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """
    label_dict = {label_list[0]: 0, label_list[1]: 0}
    smooth = 1  # smoothing factor
    logprob = {}
    # TODO: add your code here
    for label in training_data:
        label_dict[label['label']] += 1

    logprob['2016'] = math.log(
        (label_dict['2016'] + smooth) / (label_dict['2016'] + label_dict['2020'] + smooth + smooth))
    logprob['2020'] = math.log(
        (label_dict['2020'] + smooth) / (label_dict['2016'] + label_dict['2020'] + smooth + smooth))
    return logprob


# Needs modifications
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """
    total_word_count = 0
    wordcount = {}
    oov_count = 0
    for data_dict in training_data:
        if data_dict['label'] == label:
            bow = data_dict['bow']
            for word in bow.keys():
                if word != 'None':
                    total_word_count += bow[word]
                    if word not in wordcount:
                        wordcount[word] = bow[word]
                    else:
                        wordcount[word] += bow[word]
                else:
                    oov_count += bow[word]
    smooth = 1  # smoothing factor
    total_word_count += oov_count
    word_prob = {}
    # TODO: add your code here
    for key in vocab:
        if key not in wordcount:
            wordcount[key] = 0
        word_prob[key] = math.log((wordcount[key] + smooth) / (total_word_count + smooth * (len(vocab) + 1)))
    word_prob['None'] = math.log((oov_count + smooth) / (total_word_count + smooth * (len(vocab) + 1)))
    return word_prob


##################################################################################
# Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    label_list = [f for f in os.listdir(training_directory) if not f.startswith('.')]  # ignore hidden files
    # TODO: add your code here

    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)
    retval = {'vocabulary': vocab,
              'log prior': prior(training_data, label_list),
              'log p(w|y=2016)': p_word_given_label(vocab, training_data, '2016'),
              'log p(w|y=2020)': p_word_given_label(vocab, training_data, '2020')}

    return retval


# Needs modifications
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    # TODO: add your code here
    total2016 = model['log prior']['2016']
    total2020 = model['log prior']['2020']
    with open(filepath) as doc:
        for word in doc:
            word = word.strip()
            if word in model['log p(w|y=2020)']:
                total2020 += model['log p(w|y=2020)'][word]
            else:
                total2020 += model['log p(w|y=2020)']['None']
            if word in model['log p(w|y=2016)']:
                total2016 += model['log p(w|y=2016)'][word]
            else:
                total2016 += model['log p(w|y=2016)']['None']

    if total2016 >= total2020:
        result = '2016'
    else:
        result = '2020'

    retval = {'log p(y=2020|x)': total2020,
              'log p(y=2016|x)': total2016,
              'predicted y': result}

    return retval
