
def get_dict():
    with open("Corel-5k/corel5k_vocabulary.txt") as f:
        word_list = f.readlines()

    dictionary = {}

    for idx, word in enumerate(word_list):
        dictionary[word.strip()] = idx
    
    return dictionary
