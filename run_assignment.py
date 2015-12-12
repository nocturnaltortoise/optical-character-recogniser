import scipy.linalg as la
from ErrorCorrector import *

# Can we get proper nouns from the test labels somehow?
# possibly words that aren't in the dictionary and are capitalised - just capitalised words would catch the start
# of sentences, but scanning the dictionary is expensive - although in this case it could be done with an if in set

# npy and dat files will actually be in the same directory as the code, not in data/
page1 = np.load('data/train1.npy')
page2 = np.load('data/train2.npy')
page3 = np.load('data/train3.npy')
page4 = np.load('data/train4.npy')

test1 = np.load('data/test1.npy')
test1_2 = np.load('data/test1.2.npy')
test1_3 = np.load('data/test1.3.npy')
test1_4 = np.load('data/test1.4.npy')

test2 = np.load('data/test2.npy')
test2_2 = np.load('data/test2.2.npy')
test2_3 = np.load('data/test2.3.npy')
test2_4 = np.load('data/test2.4.npy')

page1_boxes = np.loadtxt('data/train1.dat',
                         dtype={'names': ('labels', 'left', 'bottom', 'right', 'top', 'word_end'),
                                'formats': ('S1', np.int, np.int, np.int, np.int, np.int)})
page2_boxes = np.loadtxt('data/train2.dat',
                         dtype={'names': ('labels', 'left', 'bottom', 'right', 'top', 'word_end'),
                                'formats': ('S1', np.int, np.int, np.int, np.int, np.int)})
page3_boxes = np.loadtxt('data/train3.dat',
                         dtype={'names': ('labels', 'left', 'bottom', 'right', 'top', 'word_end'),
                                'formats': ('S1', np.int, np.int, np.int, np.int, np.int)})
page4_boxes = np.loadtxt('data/train4.dat',
                         dtype={'names': ('labels', 'left', 'bottom', 'right', 'top', 'word_end'),
                                'formats': ('S1', np.int, np.int, np.int, np.int, np.int)})
test1_boxes = np.loadtxt('data/test1.dat',
                         dtype={'names': ('labels', 'left', 'bottom', 'right', 'top', 'word_end'),
                                'formats': ('S1', np.int, np.int, np.int, np.int, np.int)})
test2_boxes = np.loadtxt('data/test2.dat',
                         dtype={'names': ('labels', 'left', 'bottom', 'right', 'top', 'word_end'),
                                'formats': ('S1', np.int, np.int, np.int, np.int, np.int)})


def find_words(chars, boxes):

    """Finds the words in an array of characters using the bounding box data."""
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
        # padded_image.fill(255)
        # filling with white actually decreases the performance considerably, for some reason

        padded_image[0:bottom-top, 0:right-left] = image
        result_array[i] = np.ravel(padded_image)

    return result_array


def prepare_images(page_data, page_boxes, max_width, max_height):

    """Creates an empty matrix of the correct size, uses generate_padded_images to get the flattened feature vectors.
    @param page_data: the whole page of characters to be prepared.
    @param page_boxes: the bounding boxes for that page.
    @param max_width: the maximum width of all of the testing and training data.
    @param max_height: the maximum height of all of the testing and training data.
    """

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


def run_pca(num_features, test_data, train_data):

    """Takes the number of axes and some testing and training data and uses PCA to reduce the dimensions of that data to
    the number of features specified.

    Args
    @param num_features: Number of features to select with PCA
    @param test_data: Test data to be reduced.
    @param train_data: Train data to be reduced.
    """

    eigenletters = generate_eigenletters(train_data, num_features)
    # try:
    #     # try and load the train data from a file
    #     pcatrain_data = np.load('aca14st_pcatraindata.npy')
    # except IOError:
        # if file doesn't exist, run dimensionality reduction on it and save the result
    pcatrain_data = np.dot((train_data - np.mean(train_data)), eigenletters)
        # np.save("aca14st_pcatraindata", pcatrain_data)

    pcatest_data = np.dot((test_data - np.mean(train_data)), eigenletters)

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

    dot_product = np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test * test, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))

    cosine_dist = dot_product/np.outer(modtest, modtrain.transpose())
    nearest = np.argmax(cosine_dist, axis=1)

    labels = train_data_labels[nearest]
    score = (100.0 * sum(test_data_labels[:] == labels))/labels.shape[0]
    return score, labels


def count_correct(correct_labels, corrected_words):

    """Processes a list of words and counts the number of characters matching another list of characters.
    @param correct_labels: Labels from the test data.
    @param corrected_words: List of words "corrected" by error correction.
    """

    correct_count = 0
    for i in xrange(len(corrected_words)):
        corrected_words_characters = "".join(corrected_words)

    for i in xrange(len(corrected_words_characters)):
        if corrected_words_characters[i] == correct_labels[i]:
            correct_count += 1

    return 100.0 * correct_count / len(corrected_words_characters)


def find_max_dimensions(test_boxes):

    """Finds the maximum dimensions of all of the data (both test and training).
    @param test_boxes: The bounding boxes of the testing data."""

    max_train_width, max_train_height = max(find_dimensions(page1_boxes),
                                            find_dimensions(page2_boxes),
                                            find_dimensions(page3_boxes),
                                            find_dimensions(page4_boxes))

    max_test_width, max_test_height = find_dimensions(test_boxes)

    max_width, max_height = max((max_train_width, max_train_height), (max_test_width, max_test_height))
    return max_width, max_height


def evaluate_results(test_page_prepared, num_features, test_boxes, train_data, train_labels, test_labels):

    """Given the labels, number of PCA features, testing and training data,
    runs PCA, the classifier, and error correction, and prints out results.
    @param test_page_prepared: The test data to use.
    @param num_features: Number of dimensions to reduce the data to with PCA.
    @param test_boxes: Bounding boxes for the test data.
    @param train_data: The training data to use.
    @param train_labels: The labels of the training data.
    @param test_labels: The labels of the testing data.
    """

    pca_results = run_pca(num_features, test_page_prepared, train_data)
    pcatraindata = pca_results[0]
    pcatestdata = pca_results[1]
    classify_score, labelled_chars = classify(pcatraindata, train_labels, pcatestdata, test_labels)
    predicted_words = find_words(labelled_chars, test_boxes)
    corrected_words = run_error_correction(test_boxes, predicted_words)

    # uncomment these to see the scores and predicted words before error correction
    # print classify_score
    # print predicted_words
    print count_correct(test_labels, corrected_words)
    print corrected_words


def run_error_correction(test_boxes, predicted_words):

    """Attempts to correct errors on a list of words.
    @param test_boxes: The bounding boxes of the test data.
    @param predicted_words: A list of words created from the labels the classifier identified, and the bounding boxes.
    """
    correct_words = find_words(test_boxes["labels"], test_boxes)

    # count_big.txt is a list of words created from stitching together several books from Project Gutenberg,
    # courtesy of Peter Norvig.
    error_corrector = ErrorCorrector('data/count_big.txt')
    # run the error correction for 1 edit distance
    corrected_words = error_corrector.correct_words(predicted_words, correct_words, 1)

    # for every remaining incorrect word, run the error corrector again, with a higher edit distance
    for i in xrange(len(corrected_words)):
        if corrected_words[i] != correct_words[i]:
            corrected_words[i] = error_corrector.correct_words(corrected_words[i], correct_words, 2)

    return corrected_words


def run(test_page, test_boxes, num_features):

    width, height = find_max_dimensions(test_boxes)

    page1_prepared = prepare_images(page1, page1_boxes, width, height)
    page2_prepared = prepare_images(page2, page2_boxes, width, height)
    page3_prepared = prepare_images(page3, page3_boxes, width, height)
    page4_prepared = prepare_images(page4, page4_boxes, width, height)

    test_page_prepared = prepare_images(test_page, test_boxes, width, height)

    images = np.vstack((page1_prepared, page2_prepared, page3_prepared, page4_prepared))

    train_labels = np.hstack((page1_boxes["labels"],
                              page2_boxes["labels"],
                              page3_boxes["labels"],
                              page4_boxes["labels"]))

    test_labels = test_boxes["labels"]

    evaluate_results(test_page_prepared, num_features, test_boxes, images, train_labels, test_labels)


print "Trial 1: No noise: "
run(test1, test1_boxes, 40)
run(test2, test2_boxes, 40)

print "Trial 2: Noisy Data: "
run(test1_2, test1_boxes, 40)
run(test1_3, test1_boxes, 40)
run(test1_4, test1_boxes, 40)
run(test2, test2_boxes, 40)
run(test2_2, test2_boxes, 40)
run(test2_3, test2_boxes, 40)
run(test2_4, test2_boxes, 40)

print "Trial 3: 10 Features: "

print "No noise: "
run(test1, test1_boxes, 10)
run(test2, test2_boxes, 10)

print "Noisy Data: "
run(test1_2, test1_boxes, 10)
run(test1_3, test1_boxes, 10)
run(test1_4, test1_boxes, 10)
run(test2, test2_boxes, 10)
run(test2_2, test2_boxes, 10)
run(test2_3, test2_boxes, 10)
run(test2_4, test2_boxes, 10)










