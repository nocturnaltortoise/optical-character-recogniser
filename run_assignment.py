import numpy as np
import scipy.linalg as la
from ErrorCorrector import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

# npy and dat files will actually be in the same directory as the code, not in data/
page1 = np.load('data/train1.npy')
page2 = np.load('data/train2.npy')
page3 = np.load('data/train3.npy')
page4 = np.load('data/train4.npy')

test1 = np.load('data/test1.npy')
test2 = np.load('data/test2.npy')

page1_boxes = np.loadtxt('data/train1.dat', dtype={'names': ('labels', 'left', 'bottom', 'right', 'top', 'word_end'),
                                                   'formats': ('S1', np.int, np.int, np.int, np.int, np.int)})
page2_boxes = np.loadtxt('data/train2.dat', dtype={'names': ('labels', 'left', 'bottom', 'right', 'top', 'word_end'),
                                                   'formats': ('S1', np.int, np.int, np.int, np.int, np.int)})
page3_boxes = np.loadtxt('data/train3.dat', dtype={'names': ('labels', 'left', 'bottom', 'right', 'top', 'word_end'),
                                                   'formats': ('S1', np.int, np.int, np.int, np.int, np.int)})
page4_boxes = np.loadtxt('data/train4.dat', dtype={'names': ('labels', 'left', 'bottom', 'right', 'top', 'word_end'),
                                                   'formats': ('S1', np.int, np.int, np.int, np.int, np.int)})
test1_boxes = np.loadtxt('data/test1.dat', dtype={'names': ('labels', 'left', 'bottom', 'right', 'top', 'word_end'),
                                                  'formats': ('S1', np.int, np.int, np.int, np.int, np.int)})
test2_boxes = np.loadtxt('data/test2.dat', dtype={'names': ('labels', 'left', 'bottom', 'right', 'top', 'word_end'),
                                                  'formats': ('S1', np.int, np.int, np.int, np.int, np.int)})


def find_words(chars, boxes):
    words = []
    for i in xrange(boxes.shape[0]):
        words.append(chars[i])
        if int(boxes[i]["word_end"]) == 1:
            words.append(" ")

    words = "".join(words).split(" ")
    return words


def find_dimensions(boxes):

    """Find the maximum height and width of any character from their bounding boxes.

    Args
    @param boxes: the bounding boxes to find dimensions for.
    """

    current_max_width = 0
    current_max_height = 0
    for i in xrange(boxes.shape[0]):
        if (boxes[i]["right"] - boxes[i]["left"]) > current_max_width:
            current_max_width = boxes[i]["right"] - boxes[i]["left"]

        if (boxes[i]["top"] - boxes[i]["bottom"]) > current_max_height:
            current_max_height = boxes[i]["top"] - boxes[i]["bottom"]

    return current_max_width, current_max_height


def generate_padded_images(input_data, boxes, result_array, max_width, max_height):

    """Pad out and reshape (ravel) images to fit an array of equal sized 1-d feature vectors.

    Args
    @param input_data: the images to be separated and padded
    @param boxes: the bounding boxes for each of the characters within the input data
    @param result_array: the padded and flattened feature vectors for each image
    @param max_width: the maximum width of any bounding box
    @param max_height: the maximum height of any bounding box

    """
    for i in xrange(boxes.shape[0]):
        left = boxes[i]["left"]
        right = boxes[i]["right"]
        bottom = input_data.shape[0] - boxes[i]["bottom"]
        top = input_data.shape[0] - boxes[i]["top"]

        image = np.reshape(input_data[top:bottom, left:right], (top-bottom, right-left))

        padded_image = np.zeros((max_height, max_width), dtype=np.int)
        padded_image.fill(255)

        padded_image[0:bottom-top, 0:right-left] = image
        result_array[i] = np.ravel(padded_image)

    return result_array


def prepare_images(page_data, page_boxes, max_width, max_height):
    empty_images = np.zeros((page_boxes.shape[0], max_width * max_height), dtype=np.int)
    page_images = generate_padded_images(page_data, page_boxes, empty_images, max_width, max_height)
    return page_images


def generate_eigenletters(train_data, feature_number):

    """Generate eigenvalues from covariance matrix of training data.

    Args
    @param train_data: the padded feature vectors of all the character images
    @param feature_number: number of PCA axes to use
    """
    covx = np.cov(train_data, rowvar=0)
    cov_size = covx.shape[0]
    eigenvalues, eigenvectors = la.eigh(covx, eigvals=(cov_size - feature_number, cov_size - 1))
    eigenvectors = np.fliplr(eigenvectors)
    return eigenvectors


def run_pca(num_features):
    print "Running PCA..."
    eigenletters = generate_eigenletters(images, num_features)
    pcatrain_data = np.dot((images - np.mean(images)), eigenletters)
    pcatest_data = np.dot((test_page1_prepared - np.mean(images)), eigenletters)

    return pcatrain_data, pcatest_data


def classify(train, train_data_labels, test, test_data_labels, features=None):

    """Nearest neighbour classification implementation
        by Jon Barker (http://staffwww.dcs.shef.ac.uk/people/J.Barker/)

    @param train: the training data for the classifier
    @param train_data_labels: the labels for the training data characters
    @param test: the testing data to compare with the training data
    @param test_data_labels: the labels of the testing data characters
    @param features: the number of features to classify on, if features=None then all the features are used

    returns: (score, labels) - a percentage correct (predicted labels compared to actual) and the classifier's labels
    """
    # Use all features if no feature parameter has been supplied
    if features is None:
        features = np.arange(0, train.shape[1])

    # Select the desired features from the training and test data
    train = train[:, features]
    test = test[:, features]

    # Super compact implementation of nearest neighbour
    x = np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test * test, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))

    dist = x/np.outer(modtest, modtrain.transpose())  # cosine distance
    nearest = np.argmax(dist, axis=1)

    labels = train_data_labels[nearest]
    score = (100.0 * sum(test_data_labels[:] == labels))/labels.shape[0]
    return score, labels


max_train_width, max_train_height = max(find_dimensions(page1_boxes),
                                        find_dimensions(page2_boxes),
                                        find_dimensions(page3_boxes),
                                        find_dimensions(page4_boxes))

max_test_width, max_test_height = find_dimensions(test1_boxes)

max_width, max_height = max((max_train_width, max_train_height), (max_test_width, max_test_height))

page1_prepared = prepare_images(page1, page1_boxes, max_width, max_height)
page2_prepared = prepare_images(page2, page2_boxes, max_width, max_height)
page3_prepared = prepare_images(page3, page3_boxes, max_width, max_height)
page4_prepared = prepare_images(page4, page4_boxes, max_width, max_height)

images = np.vstack((page1_prepared, page2_prepared, page3_prepared, page4_prepared))

test_page1_prepared = prepare_images(test1, test1_boxes, max_width, max_height)
# test_page2_prepared = prepare_images(test2, test2_boxes, max_width, max_height)

train_labels = np.hstack((page1_boxes["labels"], page2_boxes["labels"], page3_boxes["labels"], page4_boxes["labels"]))
test_labels = test1_boxes["labels"]


num_features = 10
pca_results = run_pca(num_features)
pcatraindata = pca_results[0]
pcatestdata = pca_results[1]
classify_score, labelled_chars = classify(pcatraindata, train_labels, pcatestdata, test_labels)

predicted_words = find_words(labelled_chars, test1_boxes)
correct_words = find_words(test1_boxes["labels"], test1_boxes)

print classify_score
print predicted_words
print "Correcting errors..."
error_corrector = ErrorCorrector('data/count_1w.txt')
corrected_words = error_corrector.correct_words(predicted_words, correct_words)
print corrected_words


def count_correct():
    correct_count = 0
    for i in xrange(len(corrected_words)):
        if corrected_words[i] == correct_words[i]:
            correct_count += 1

    return 100.0 * correct_count / len(correct_words)

print count_correct()
