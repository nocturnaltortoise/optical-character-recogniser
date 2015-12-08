import numpy as np


class ErrorCorrector:

    def __init__(self, frequency_dictionary_path):
        self.common_words = np.loadtxt(frequency_dictionary_path, dtype={'names': ('word', 'frequency'), 'formats': ('S20', np.int)})
        # self.dictionary = np.loadtxt(standard_dictionary_path, dtype=
        self.dictionary = self.common_words["word"]
        # not sure whether using two different dictionaries helped - error correction isn't more effective

    @staticmethod
    def __calculate_edit_distance(word1, word2):

        """Work out the number of operations to get from word1 to word2"""

        edit_distance = 0

        # if len(word1) != len(word2):
        #     edit_distance += abs(len(word1) - len(word2))

        min_word_length = min(len(word1), len(word2))

        for i in xrange(min_word_length):
            if word1[i] != word2[i]:
                # edit_distance += abs(ord(word1[i]) - ord(word2[i]))
                edit_distance += 1

        return edit_distance

    def __get_nearest_words(self, word1, dictionary):

        """Given a word, find the nearest words (within 3 edits) by edit distance."""
        # need to prioritise edit distance 1 words before 2 and 3, and only then consider frequency
        # within a set of equal edit distance words
        nearest_words = []
        for word in dictionary:
            edit_distance = self.__calculate_edit_distance(word1.lower(), str(word))
            if edit_distance <= 1 and len(str(word)) == len(word1):  # edit distance threshold - most mistakes aren't more than three characters
                # nearest_words.append((word, edit_distance))   # zipping edit_distances with the words so we can sort
                nearest_words.append(str(word))
        # nearest_words = [word for word in dictionary if calculate_edit_distance(word1,word) <= 3] #this kinda works
        return nearest_words

    @staticmethod
    def __get_highest_freq_word(words, dictionary):

        """Given a list of words, return the one with the highest frequency in the English language, or at least in my list of frequencies."""
        # consider using collocates - i.e. nearby words as well, so frequencies can be considered in terms of word-pairs
        for freq_word in dictionary:
            for word in words:
                if str(word) == str(freq_word):
                    return freq_word
                else:
                    return word


    # print get_highest_freq_word(get_nearest_words("tbe", dictionary), dictionary)

    def correct_words(self, predicted_words, correct_words):
        # incorrect_words = set(correct_words) - set(predicted_words)
        corrected_words = []

        for word in predicted_words:
            if word not in correct_words:
                near_words = self.__get_nearest_words(word.lower(), self.dictionary)
                if len(near_words) != 0:
                    corrected_words.append(self.__get_highest_freq_word(near_words, self.dictionary))
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        return corrected_words

    # for word in incorrect_words
        # check each character with the correct word's characters
            # for each incorrect character, try correcting to a commonly mistaken character e -> c, f -> t, l -> i etc
                # check characters again, try all possible combinations?
    # this might work, but it's very expensive - O(n!) sort of expensive
