import re
import random


class Batch:

    def __init__(self, fname, hparams, mode, shuffle=True):
        self.fname = fname
        self.batch_size = int(hparams['global']['batch_size'])
        self.C = float(hparams[mode]['C'])
        self.shuffle = shuffle
        
        # patterns
        self.sentiment_tag = re.compile(
            r'\w/[p0nt]'
        )
        
        self._read_data()

    def __call__(self):
        if self.shuffle:
            random.shuffle(self._data)
        for batch in self._make_batch():
            yield batch

    def _read_data(self):
        with open(self.fname, 'r') as file:
            self._data = [
                sentence.strip() for sentence in file.readlines()
                ]
            file.close()

    def _get_pw(self, k, m, i, n):
        k += 1
        i += 1
        
        if i == k:
            pw = 1
        
        elif i < (k + m):
            pw = 1 - ((k + m - i) / self.C)

        elif (k + m) <= i and i <= n:
            pw = 1 - ((i - k) / self.C)
        
        else:
            pw = 0

        return round(pw, ndigits=3) if pw > 0 else 0

    def _parse_sentence(self, sentence):
        """
        Parse sentence to get sentence, target, polarity_one_hot, distance and position weight
        @sentence: unsplitted sentence
        """
        target_sequence = re.findall(self.sentiment_tag, sentence)
        polarity = target_sequence[0][-1] if target_sequence else -1
        # sentence = sentence.split()
        sentence_withlabel = [x for x in list(sentence) if x.strip() !=""]
        sentence = [x for x in list(re.sub(r'/[p0nt]',"",sentence)) if x.strip() !=""]

        # get polarity
        if polarity == 'p':
            polarity_one_hot = [1, 0, 0]

        elif polarity == '0':
            polarity_one_hot = [0, 0, 1]

        elif polarity == 'n':
            polarity_one_hot = [0, 1, 0]
        
        elif polarity == 't':
            polarity_one_hot = [1,1,1]

        else:
            polarity_one_hot = [0, 0, 0]

        # target_sequence.insert(0, '<BOS>')
        # target_sequence.append('<EOS>')

        # sentence.insert(0, '<BOS>')
        # sentence.append('<EOS>')

        # get position weight
        target_length = len(target_sequence)
        sentence_length = len(sentence)
        target_roller, target_index = 0, 0
        got_it = False
        for word in sentence_withlabel:
            if re.match(r'[p0nt]', word) and not got_it and target_roller>1 and sentence_withlabel[target_roller-1] =="/":
            # if re.match(r'.+/[p0nt]', word) and not got_it:
            #     target_index = target_roller
                target_index = target_roller-2
                got_it = True

            else:
                target_roller += 1

        pw = [
            self._get_pw(target_index, target_length, idx, sentence_length)
            for idx in range(0, sentence_length)
        ]

        target_sequence = [
            re.sub(r'/[p0nt]', '', target) for target in target_sequence
        ]
        # sentence = [
        #     re.sub(r'/[p0n]', '', word) for word in sentence
        #     ]
        # print(sentence, target_sequence, pw, polarity_one_hot)
        return sentence, target_sequence, pw, polarity_one_hot

    def _make_batch(self):
        start_index, end_index = 0, self.batch_size
        batch_nums = (len(self._data)-1) // self.batch_size +1
        # print("++++++++++++++++++",batch_nums,len(self._data))
        # print(self._data)
        for _ in range(batch_nums):
            temp_data = self._data[start_index:end_index]

            # parse sentence
            res = list(map(
                self._parse_sentence, 
                temp_data
                ))
            
            res = list(filter(
                lambda s: s[3] != [0, 0, 0],
                res
            ))

            yield res
            start_index, end_index = end_index, end_index + self.batch_size