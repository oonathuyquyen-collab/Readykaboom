"""Vocab class matching step27_prepare_bert4eth so the pickled vocab.pkl loads."""
class Vocab:
    def __init__(self):
        self.token2id = {"[PAD]": 0, "[UNK]": 1}
        self.id2token = {0: "[PAD]", 1: "[UNK]"}
    def add(self, token):
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
    def get_id(self, token):
        return self.token2id.get(token, 1)
