import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    """
    Tách câu thành các mảng các token/word
    Token là một từ hoặc dấu câu hoặc số
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = tìm từ gốc
    ví dụ:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    Trả về một vector:
    -0 ở vị trí của từ không có trong câu
    -1 ở vị trí của từ có trong câu
    Ví dụ:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # Stem các từ trong câu
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Khởi tạo bag
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
