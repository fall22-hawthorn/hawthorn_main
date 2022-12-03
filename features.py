import numpy as np
import pandas as pd
import language_tool_python
import readability
import enchant
from enchant.checker import SpellChecker
from collections import OrderedDict
import nltk
from nltk.tokenize import word_tokenize


class FeatureGenerator:
    def __init__(self):
        self.spell_checker = SpellChecker('en_US')
        self.enchant_dict = enchant.Dict("en_US")

        common_words_1k_filename = './1-1000.txt'
        with open(common_words_1k_filename) as f:
            self.common_words_1k = set(x.strip() for x in f.readlines())

        common_words_filename_10k = './google-10000-english-no-swears.txt'
        with open(common_words_filename_10k) as f:
            self.common_words_10k = set(x.strip() for x in f.readlines())

        # make sure common_words are a subset as well
        self.common_words_10k.update(self.common_words_1k)

        profanity_filename = './profanity.txt'
        with open(profanity_filename) as f:
            self.profanity_set = set(x.strip() for x in f.readlines())

        words_freq_filename = "./count_1w.txt"
        self.words_freq = pd.read_csv(words_freq_filename,
                             names=['word', 'freq'],
                             sep='\t',
                             header=None,
                             dtype={'word': str, 'freq': int},
                             keep_default_na=False,
                             na_values=[''])

        self.words_freq = self.words_freq.set_index('word')

        self.language_tool = language_tool_python.LanguageTool('en-US')


    def text_preprocess(self, text: str):
        '''
        Transform text to be processed by readability

        :param text: input text

        :return: str preprocessed text
        '''
        text = text.strip()

        # new paragraph starts with \n\n
        # readability also requires each sentence to end with \n
        paragraphs = [p.strip()\
                        .replace('. ', '.\n')\
                        .replace('? ', '?\n')\
                        .replace('! ', '!\n') for p in text.split('\n\n')]

        return "\n\n".join(paragraphs)


    def misspelled_count(self, text: str):
        '''
        Get count out misspelled words by enchant SpellChecker

        :param text: input text

        :return: count of misspelled words in text
        '''
        self.spell_checker.set_text(text)
        return len(list(self.spell_checker))

    def flatten_readability(self, r: OrderedDict):
        '''
        Flatten readability output by adding prefixes

        :param r: OrderedDict of readability output

        :return: dict
        '''
        out = {}
        for k, group in r.items():
            prefix = {'readability grades': '',
                      'sentence info': '',
                      'word usage': 'wu_',
                      'sentence beginnings': 'sb_'}[k]
            for var_name, value in group.items():
                out[prefix + var_name] = value
        return out

    def get_noncommon_words_count(self, text, common_words_dict):
    # first, tokenize the text
    # second, iterate over tokens and see whether
    # a. it is a word
    # b. not in common_words
    # c. correctly spelled
    # d. does not have underscore '_'

        def is_noncommon_word(w):
            return  len(w) > 2 and\
                    w not in common_words_dict and\
                    '_' not in w and\
                    self.enchant_dict.check(w)

        return sum(is_noncommon_word(x) for x in word_tokenize(text.lower()))


    def get_noncommon_words_count_1k(self, text):
        return self.get_noncommon_words_count(text, self.common_words_1k)

    def get_noncommon_words_count_10k(self, text):
        return self.get_noncommon_words_count(text, self.common_words_10k)



    def get_profanity_count(self, text):
        return sum(x in self.profanity_set for x in word_tokenize(text.lower()))

    def get_uncommon_words_counts(self, text):
        word_freq_thresholds = np.array([1e8, 1e7, 1e6, 1e5], dtype=int)
        counts = np.zeros(len(word_freq_thresholds), dtype=int)
        for w in word_tokenize(text):
            if len(w) <  3 or '_' in w or not self.enchant_dict.check(w) or w not in self.words_freq.index:
                continue
            w_freq = self.words_freq.loc[w].values[0]
            counts += (word_freq_thresholds > w_freq)
        return counts



    # I have commented out categories giving 0 columns in train dataset
    # SEMANTICS is mostly 0, we drop it as well
    LT_categories = ['CASING',
                     #'COLLOQUIALISMS',
                     'COMPOUNDING',
                     'CONFUSED_WORDS',
                     #'FALSE_FRIENDS',
                     #'GENDER_NEUTRALITY',
                     'GRAMMAR',
                     'MISC',
                     'PUNCTUATION',
                     'REDUNDANCY',
                     #'REGIONALISMS',
                     #'REPETITIONS',
                     #'REPETITIONS_STYLE',
                     #'SEMANTICS',
                     'STYLE',
                     'TYPOGRAPHY',
                     'TYPOS',
                     'TOTAL',
                    ]
    def get_LT_features(self, text):
        '''
        Generates LanguateTool features: each category count and a total number
        '''
        matches = self.language_tool.check(text)
        cat_counts = [sum(m.category == cat for m in matches) for cat in self.LT_categories[:-1]]
        return cat_counts + [len(matches)]


    def generate_features(self, df: pd.DataFrame):
        '''
        Generate features from a dataframe with `full_text` column containing english text

        :param df: input dataframe

        :return: pd.DataFrame with features and possibly updated `full_text` column
        '''

        res_df = df.copy()
        #res_df['full_text'] = res_df['full_text'].apply(self.text_preprocess)



        features_df = res_df[['full_text']].apply(lambda row:
                                                     self.flatten_readability(
                                                         readability.getmeasures(
                                                             self.text_preprocess(row[0]), lang='en')),
                                                 axis='columns',
                                                 result_type='expand')

        features_df['text_len'] = res_df['full_text'].apply(lambda x: len(x))

        features_df['misspelled'] = res_df['full_text'].apply(self.misspelled_count)
        features_df['noncommon_words_1k'] = res_df['full_text'].apply(self.get_noncommon_words_count_1k)
        features_df['noncommon_words_10k'] = res_df['full_text'].apply(self.get_noncommon_words_count_10k)
        features_df['profanity_count'] = res_df['full_text'].apply(self.get_profanity_count)
        features_df[['uwc1e8', 'uwc1e7', 'uwc1e6', 'uwc1e5']] =\
            res_df[['full_text']].apply(lambda x: self.get_uncommon_words_counts(x[0]),
                                        axis='columns',
                                        result_type='expand')

        # Generate ratio features
        words_ratio_features = ['wordtypes',
                                'long_words',
                                'complex_words',
                                'complex_words_dc',
                                'wu_tobeverb',
                                'wu_auxverb',
                                'wu_conjunction',
                                'wu_pronoun',
                                'wu_preposition',
                                'wu_nominalization',
                                'misspelled',
                                'noncommon_words_1k',
                                'noncommon_words_10k',
                                'uwc1e8',
                                'uwc1e7',
                                'uwc1e6',
                                'uwc1e5',
                               ]
        features_df[[x + "_ratio" for x in words_ratio_features]] = features_df[words_ratio_features]\
                                                                        .div(features_df['words'], axis=0)
        sentences_ratio_features = ['sb_pronoun',
                                    'sb_interrogative',
                                    'sb_article',
                                    'sb_subordination',
                                    'sb_conjunction',
                                    'sb_preposition',
                                   ]
        features_df[[x + "_ratio" for x in sentences_ratio_features]] = features_df[sentences_ratio_features]\
                                                                            .div(features_df['sentences'], axis=0)


        features_df[['LT_' + x for x in self.LT_categories]] = res_df[['full_text']]\
                                                            .apply(lambda x: self.get_LT_features(x[0]),
                                                                   axis=1,
                                                                   result_type='expand')

        features_df[['LT_' + x + '_ratio' for x in self.LT_categories]] = features_df[['LT_' + x for x in self.LT_categories]]\
                                                                        .div(features_df['words'], axis=0)

        features_df = features_df.sort_index(axis=1)
        return pd.concat([res_df, features_df], axis='columns')
