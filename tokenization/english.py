import spacy
import re


class Eng_Tokenization_NLP(object):
    """
    English tokenization class. Needs en_core_web_sm installed first.
    """

    def __init__(self, nlp=None):
        if nlp is None:
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = nlp
        self.tokenized = []
        self.list_pos = []
        self.pos_tokens = {}
        self.target = ["NOUN", "VERB", "PROPN", "ADJ", "ADP", "AUX", "PRON", "DET"]

    def invalidNewsForChecking(self):
        print("Not a valid news to be checked")

    def validNewsForChecking(self):
        print("Valid news to be checked")

    def tokenizationProcess(self, word_list):

        for word in word_list:
            self.tokenized.append(word)

        combined_words = " ".join(self.tokenized).replace("-", "_")
        doc = self.nlp(re.sub("[^A-Za-z0-9_]+", " ", combined_words))
        for tokens in doc:
            # print(f"Parts of speech in token: {tokens.pos_}")
            current_word = tokens.text.replace("_", "-")
            if tokens.pos_ in self.target:
                elements = f"{tokens.pos_} {current_word}"
                # print(elements)
            self.list_pos.append(tokens.pos_)

            if tokens.pos_ not in self.pos_tokens.keys():
                self.pos_tokens[tokens.pos_] = [
                    current_word,
                ]
            else:
                self.pos_tokens[tokens.pos_].append(current_word)
