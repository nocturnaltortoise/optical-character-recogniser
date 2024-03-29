import numpy as np


class ErrorCorrector:

    def __init__(self, frequency_dictionary_path):
        self.common_words = np.loadtxt(frequency_dictionary_path,
                                       dtype={'names': ('word', 'frequency'),
                                              'formats': ('S20', np.int)})
        # sort the dictionary by frequency, in descending order
        self.common_words = np.sort(self.common_words, order='frequency')[::-1]
        self.dictionary = self.common_words["word"]

    @staticmethod
    def __calculate_edit_distance(word1, word2):

        """Work out the number of operations to get from word1 to word2"""

        edit_distance = 0
        min_word_length = min(len(word1), len(word2))

        for i in xrange(min_word_length):
            if word1[i] != word2[i]:
                edit_distance += 1

        return edit_distance

    def __get_nearest_words(self, word1, dictionary, edit_threshold):

        """Given a word, find the nearest words (within 1 or 2 edits) by edit distance."""
        nearest_words = []
        for word in dictionary:
            edit_distance = self.__calculate_edit_distance(word1.lower(), str(word))
            if edit_distance <= edit_threshold and len(str(word)) == len(word1):
                # only need to consider same length words as we know where the words end.
                nearest_words.append(str(word))
        return nearest_words

    @staticmethod
    def __get_highest_freq_word(words, dictionary):

        """Given a list of words, return the one with the highest frequency in the dictionary."""
        for freq_word in dictionary:
            for word in words:
                if str(word) == str(freq_word):
                    return freq_word
                else:
                    return word

    def correct_words(self, predicted_words, correct_words, edit_distance):

        """Correct either a single word or a list of words.
        @param predicted_words: Words from the classifier's labels.
        @param correct_words: Words from the correct labels of the test data.
        @param edit_distance: The edit distance threshold for __get_nearest_words()."""

        # if correcting only one word
        if type(predicted_words) is not list:
            if predicted_words not in correct_words:
                near_words = self.__get_nearest_words(predicted_words.lower(), self.dictionary, edit_distance)
                if len(near_words) != 0:
                    # if there are nearby words, return the highest frequency word
                    return self.__get_highest_freq_word(near_words, self.dictionary)
                else:
                    # otherwise return the original
                    return predicted_words
        else:
            # if correcting a list of words
            for i in xrange(len(predicted_words)):
                if predicted_words[i] not in correct_words:
                    near_words = self.__get_nearest_words(predicted_words[i].lower(), self.dictionary, edit_distance)
                    if len(near_words) != 0:
                        # if there are nearby words, change the word being corrected to the highest frequency word
                        predicted_words[i] = (self.__get_highest_freq_word(near_words, self.dictionary))

        return predicted_words
