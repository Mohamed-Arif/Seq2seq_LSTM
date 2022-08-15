# This python code has been referred from https://github.com/hoang-tn-nguyen/Disfl-QA-Performer/datasets.py

import json
import numpy as np
import sentencepiece as spm
import torch.utils.data as data

class DisflQA(data.Dataset):
    def __init__(self, 
                file_name='Datasets/Disfl-QA/train.json', 
                vocab_file='Datasets/Disfl-QA/spm.model', 
                src_vocab_file='Datasets/Disfl-QA/spm_disfluent.model', 
                tgt_vocab_file='Datasets/Disfl-QA/spm_original.model',
                max_len=100, return_len=False, infer=False):
        '''
        max_len: maximum output length
        return_len: return the length of sequences
        infer: a special flag for inference phase (no output is used as input for models)
        '''
        self.__input_data(file_name)
        self.vocab = spm.SentencePieceProcessor(model_file=vocab_file)
        self.src_vocab = spm.SentencePieceProcessor(model_file=src_vocab_file)
        self.tgt_vocab = spm.SentencePieceProcessor(model_file=tgt_vocab_file)
        self.max_len = max_len
        self.return_len = return_len
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input = self.src_vocab.Encode(self.data[index]['disfluent'].lower())
        output = self.tgt_vocab.Encode(self.data[index]['original'].lower())
        
        # cast to a numpy array with the same length
        input =  [self.vocab.bos_id()] + input + [self.vocab.eos_id()]
        output =  [self.vocab.bos_id()] + output + [self.vocab.eos_id()]
        np_input = np.ones(self.max_len, dtype=np.int64) * self.src_vocab.pad_id()
        np_output = np.ones(self.max_len, dtype=np.int64) * self.tgt_vocab.pad_id()
        np_input[:min(len(input), self.max_len)] = input[:min(len(input), self.max_len)]
        np_output[:min(len(output), self.max_len)] = output[:min(len(output), self.max_len)]
        
        if self.return_len:
            return (np_input, len(input)), (np_output, len(output))
        else:
            if self.infer: # During inference, the output is ignored
                # Input: np_input
                # Output: np_output
                return np_input, np_output
            else: # During training, the output is a part of the input
                # Input: (np_input, np_output)
                # Output: np_output
                return (np_input, np_output), np_output

    def __input_data(self, file_name):
        try:
            json_file = open(file_name, 'r')
            data = json.load(json_file)
            self.data = [v for k,v in data.items()]
            json_file.close()

        except Exception as e:
            print('Error occurred:', e)

if __name__ == '__main__':
    my_data = DisflQA()
    print(my_data.tgt_vocab.Decode(my_data[0][1].tolist()))

