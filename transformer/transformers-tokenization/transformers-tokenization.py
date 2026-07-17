import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2 
        self.word_to_id[self.eos_token] = 3 
        self.id_to_word[0] = self.pad_token;
        self.id_to_word[1] = self.unk_token;
        self.id_to_word[2] = self.bos_token;
        self.id_to_word[3] = self.eos_token;
        self.vocab_size = 4; 


    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        complete_vocab = set()
        for i in range(len(texts)):
            text = texts[i]; 
            text = text.lower()
            split_text = text.split()
            complete_vocab.update(split_text)

        complete_vocab = list(complete_vocab); 
        complete_vocab.sort();        
        for i in complete_vocab:
            self.word_to_id[i] = self.vocab_size; 
            self.id_to_word[self.vocab_size] = i; 
            self.vocab_size += 1; 
            
        # YOUR CODE HERE
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        text = text.lower()
        words = text.split(); 
        tokens = []
        for word in words: 
            if word not in self.word_to_id:
                word = self.unk_token; 
            tokens.append(self.word_to_id[word]);

        return tokens; 
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        text = ""
        for id in ids: 
            if id not in self.id_to_word:
                id = self.word_to_id[self.unk_token]; 
            if text != "":
                text += " "
            text += self.id_to_word[id];
        return text; 
