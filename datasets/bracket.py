class BracketTokenizer:
    def __init__(self, pad_token="<pad>", unk_token="<unk>", bos_token="<start>", eos_token="<stop>"):
        self.vocab = [bos_token, eos_token, pad_token, unk_token] + ['(', ')']
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        self.index_to_word = {idx: word for idx, word in enumerate(self.vocab)}
        self.pad_token_id = self.word_to_index[pad_token]
        self.unk_token_id = self.word_to_index[unk_token]
        self.bos_token_id = self.word_to_index[bos_token]
        self.eos_token_id = self.word_to_index[eos_token]

    def encode(self, text, max_length=None):
        """This method takes a natural text and encodes it into a sequence of token ids using the vocabulary.

        Args:
            text (str): Text to encode.
            max_length (int, optional): Maximum encoding length. Defaults to None.

        Returns:
            List[int]: List of token ids.
        """
        token_ids = [self.bos_token_id] + [self.word_to_index.get(word, self.unk_token_id) for word in text.split()] + [
            self.eos_token_id]
        if max_length:
            token_ids = token_ids[:max_length + 2] + [self.pad_token_id] * (max_length - len(token_ids) + 2)

        return token_ids

    def decode(self, sequence, skip_special_tokens=True):
        """This method takes a sequence of token ids and decodes it into a language tokens.

        Args:
            sequence (List[int]): Sequence to be decoded.
            skip_special_tokens (bool, optional): Whether to skip special tokens when decoding. Defaults to True.

        Returns:
            List[str]: List of decoded tokens.
        """
        tokens = [self.index_to_word[idx] for idx in sequence]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in [self.pad_token, self.bos_token, self.eos_token, self.unk_token]]
        return tokens
