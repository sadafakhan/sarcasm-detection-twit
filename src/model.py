import unicodedata
import json
from sklearn.linear_model import LogisticRegression
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_feature_encoding
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import numpy as np
from util import make_folders
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex

##############################################################################
# Utility function
##############################################################################

def distributions_over_labels(clf, pair: tuple) -> np.ndarray:
    """Get a classifier's distribution over labels (0, 1) for one (sentence, sentence) example. 

    Call the sklearn `predict_proba' function or force the corresponding logic when not allowed (e.g., when `voting=hard', or SVM probability=False).

    Example: 
        >>> distributions_over_labels(clf, (sent_embedding1, sent_embedding2))
        >>> array([[ 0.3,  0.7], [ 0.8,  0.2]])

    Args:
        clf: an sklearn classifier

        pair: the example of texts of form (embedding, embedding) ; predict the text more likely to be sarcastic, where `embedding' is a numpy array.

    Returns:
        scores: the distribution over the binary class labels (indices in the adaptation task), of form (label_probabilities, label_probabilities) where label_probabilities is a numpy array of 2 elements.
    """
    if (
        isinstance(clf, VotingClassifier) 
        and clf.voting == "hard"
        ):
        # Manually get probabilities :scream:
        scores = np.average(
            np.asarray(
                [estimator.predict_proba(pair) for estimator in clf.estimators_]),
            axis=0, 
            weights=clf._weights_not_none,
        )
    else:
        scores = clf.predict_proba(pair)

    return scores


##############################################################################
# Deepmoji Model to obtain embeddings
##############################################################################

deepmoji_train_vectors_filename = "src/vectors/deepmoji_train_vectors.npy"
deepmoji_dev_vectors_filename = "src/vectors/deepmoji_dev_vectors.npy"
deepmoji_test_vectors_filename = "src/vectors/deepmoji_test_vectors.npy"
deepmoji_adaptation_tweet_vectors_filename = "src/vectors/deepmoji_adaptation_tweet_vectors.npy"
deepmoji_adaptation_rephrase_vectors_filename = "src/vectors/deepmoji_adaptation_rephrase_vectors.npy"

deepmoji_train_vectors = None
deepmoji_dev_vectors = None
deepmoji_test_vectors = None
deepmoji_adaptation_tweet_vectors = None 
deepmoji_adaptation_rephrase_vectors = None 

deepmoji_polarity_train_vectors_filename = "src/vectors/deepmoji_polarity_train_vectors.npy"
deepmoji_polarity_dev_vectors_filename = "src/vectors/deepmoji_polarity_dev_vectors.npy"
deepmoji_polarity_test_vectors_filename = "src/vectors/deepmoji_polarity_test_vectors.npy"
deepmoji_adaptation_tweet_vectors_filename = "src/vectors/deepmoji_adaptation_tweet_vectors.npy"
deepmoji_adaptation_rephrase_vectors_filename = "src/vectors/deepmoji_adaptation_rephrase_vectors.npy"

deepmoji_polarity_adaptation_tweet_vectors_filename = "src/vectors/deepmoji_polarity_adaptation_tweet_vectors.npy"
deepmoji_polarity_adaptation_rephrase_vectors_filename = "src/vectors/deepmoji_polarity_adaptation_rephrase_vectors.npy"


deepmoji_polarity_train_vectors = None
deepmoji_polarity_dev_vectors = None
deepmoji_polarity_test_vectors = None
deepmoji_polarity_adaptation_tweet_vectors = None
deepmoji_polarity_adaptation_adaptation_vectors = None


class DeepMoji:
    """DeepMoji is a model with pre-trained parameters, but not a classifier.
    
    This class contains the following attributes: 
        - model/weights,
        - vocabulary,
        - tokenizer
    
    All classifiers which use DeepMoji require these attributes.
    """

    # Load VaderSentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Load vocabulary
    print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)

    # Load pretrained model
    print('Loading model from {}.'.format(PRETRAINED_PATH))
    model = torchmoji_feature_encoding(PRETRAINED_PATH)
    # print(model)

    # create tokenizer
    maxlen = 30
    tokenizer = SentenceTokenizer(vocabulary, maxlen)

    # def __init__(self, maxlen=30, batch_size=32):
    #     pass

    def get_deepmoji_vectors(self, filename, sentences, vectors, add_polarity=False):
        """Given a list of sentences and a filename, add embeddings if they aren't already contained in `vectors.'

        Args:
            filename: name of the vectors file to read from or save to (train or test).

            sentences: a list of text examples
            vectors: list of vectors (global variable) to be checked against

        Returns:
            vectors: list of deepmoji vectors updated with those for the sentences passed to this function.
        """
        print("Attempting to get " + str(len(sentences)) + " vectors")

        if not vectors:
            try:
                # First, try loading the DeepMoji vectors from file
                make_folders(filename)
                with open(filename, "rb") as deepmoji_file:
                    vectors = np.load(deepmoji_file)

            except (FileNotFoundError, IOError, TypeError) as e:  # DeepMoji file does not exist
                # Generate the vectors, putting them into a global variable
                print("File not found, creating new DeepMoji vector embeddings from " + filename)
                # Call one at a time to prevent memory error
                vectors = []
                for example in sentences:
                    deepmoji_vector = self.embedding(example)

                    if add_polarity:
                        # If we're adding the polarity, add that
                        vs = self.analyzer.polarity_scores(example)

                        #Also add emotional features from NRCLex
                        # affect_freqs = NRCLex(example).affect_frequencies

                        polarity_features = [
                            vs['pos'], # ratios
                            vs['neg'], 
                            vs['compound'], # normalized, weighted, composite 
                            
                            # affect_freqs['negative'],
                            # affect_freqs['joy'],
                            # affect_freqs['positive'],
                            # affect_freqs['trust'],
                            # affect_freqs['anger'],
                            # affect_freqs['fear'],
                            # affect_freqs['surprise'],
                            # affect_freqs['disgust'],
                            # affect_freqs['sadness'],
                            # affect_freqs['anticip'],

                            ]
                        deepmoji_vector = np.append(
                            deepmoji_vector, 
                            polarity_features,
                            )

                    vectors.append(deepmoji_vector)

                # Save the DeepMoji vectors for further use
                with open(filename, "wb") as deepmoji_file:
                    np.save(deepmoji_file, vectors)

        print("Returning" + str(len(vectors)) + " vectors")

        return vectors

    def get_train_vectors(self, train_examples):
        global deepmoji_train_vectors
        # If we haven't generated the vectors yet, generate them exactly once
        return self.get_deepmoji_vectors(deepmoji_train_vectors_filename, train_examples, deepmoji_train_vectors)

    def get_dev_vectors(self, dev_examples):
        global deepmoji_dev_vectors
        return self.get_deepmoji_vectors(deepmoji_dev_vectors_filename, dev_examples, deepmoji_dev_vectors)

    def get_test_vectors(self, test_examples):
        global deepmoji_test_vectors
        return self.get_deepmoji_vectors(deepmoji_test_vectors_filename, test_examples, deepmoji_test_vectors)

    def get_adaptation_vectors(self, examples):
        global deepmoji_adaptation_vectors
        tweets, rephrases = list(zip(*examples))
        tweets = self.get_deepmoji_vectors(
            deepmoji_adaptation_tweet_vectors_filename,
            tweets,
            deepmoji_adaptation_tweet_vectors,
        )
        rephrases = self.get_deepmoji_vectors(
            deepmoji_adaptation_rephrase_vectors_filename,
            rephrases,
            deepmoji_adaptation_rephrase_vectors,
        )
        return list(zip(tweets, rephrases))

    def get_polarity_adaptation_vectors(self, examples):
        global deepmoji_adaptation_vectors
        tweets, rephrases = list(zip(*examples))
        tweets = self.get_deepmoji_vectors(
            deepmoji_polarity_adaptation_tweet_vectors_filename,
            tweets,
            deepmoji_adaptation_tweet_vectors,
            add_polarity=True
        )
        rephrases = self.get_deepmoji_vectors(
            deepmoji_polarity_adaptation_rephrase_vectors_filename,
            rephrases,
            deepmoji_adaptation_rephrase_vectors,
            add_polarity=True
        )
        return list(zip(tweets, rephrases))

    def get_polarity_train_vectors(self, train_examples):
        global deepmoji_train_vectors
        # If we haven't generated the vectors yet, generate them exactly once
        return self.get_deepmoji_vectors(deepmoji_polarity_train_vectors_filename, train_examples,
                                         deepmoji_polarity_train_vectors, add_polarity=True)

    def get_polarity_dev_vectors(self, dev_examples):
        global deepmoji_dev_vectors
        return self.get_deepmoji_vectors(deepmoji_polarity_dev_vectors_filename, dev_examples,
                                         deepmoji_polarity_dev_vectors, add_polarity=True)

    def get_polarity_test_vectors(self, test_examples):
        global deepmoji_test_vectors
        return self.get_deepmoji_vectors(deepmoji_polarity_test_vectors_filename, test_examples,
                                         deepmoji_polarity_test_vectors, add_polarity=True)

    def embedding(self, example: str):
        """Get a deepmoji vector embedding for a single example.

        Args:
            example: a string of text

        Returns:
            an array-like object
        """
        if example == "":
            example = "The"  # We're just picking a neutral sentence in the case of an empty string, to generate an affect-neutral vector

        # convert string to deepmoji vector
        tokenized, _, _ = self.tokenizer.tokenize_sentences([example])
        vector = self.model(tokenized)[0]
        return vector


##############################################################################
# System models (classifiers)
##############################################################################

class Model:
    def __init__(self, *args, **kwargs):
        """Initialize the model, possibly with hyperparameters."""
        raise NotImplementedError

    def fit(self, train_examples, train_labels) -> None:
        """Train the model on a list of (text, label) examples.

        Not all models will be trained.

        Args:
            train_examples: list of strings representing the text examples
            train_labels: list of integers that are the labels of the train examples, 1 for `sarcastic' and 0 otherwise.
        """
        raise NotImplementedError

    def predict(self, examples, split, task) -> list:
        """Get the model predictions on a list of text examples.

        Args:
            examples: list of strings representing the text examples
        """
        raise NotImplementedError

    def classify(self, *args, **kwargs) -> int:
        """Model prediction on a single example.

        Args:
            example: a string of text

        Returns:
            an integer representing the predicted label
        """
        raise NotImplementedError

    def discriminate(self, example: tuple) -> int:
        """Model prediction on a single adaptation task example.

        Args:
            example: a tuple representing the (tweet, rephrase) pair.
        
        Returns:
            an integer representing the index of the sarcastic tweet.
        """
        raise NotImplementedError

class PredictRandom(Model):
    """Class that randomly chooses a label."""

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, train_examples, train_labels) -> None:
        """Predict random does not train."""
        pass

    def predict(self, examples, split='train', task="primary") -> list:
        """Get the model predictions on a list of text examples.

        Args:
            examples: list of strings representing the text examples

            split: a string representing what split of the data the examples are from.

        Returns: 
            the list of predictions
        """
        if task == "primary":
            prediction = self.classify
        elif task == "adaptation":
            prediction = self.discriminate

        return [prediction(example) for example in examples]

    def classify(self, *args, **kwargs) -> int:
        return np.random.choice([0, 1])
    
    def discriminate(self, example: tuple) -> int:
        """Model prediction on a single adaptation task example.

        Args:
            example: a tuple representing the (tweet, rephrase) pair.
        
        Returns:
            an integer representing the index of the sarcastic tweet.
        """
        return self.classify()

class PredictFalse(Model):
    """Class that always predicts `not sarcastic'. Because of the distribution of the train and test data, this results in about 75% correctly classified examples.
    """

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, train_examples, train_labels) -> None:
        """Predict False does not train."""
        pass

    def predict(self, examples, split='train', task="primary") -> list:
        """Get the model predictions on a list of text examples.

        Args:
            examples: list of strings representing the text examples

            split: a string representing what split of the data the examples are from.

        Returns: 
            the list of predictions
        """
        if task == "primary":
            prediction = self.classify
        elif task == "adaptation":
            prediction = self.discriminate

        return [prediction(example) for example in examples]

    def classify(self, *args, **kwargs) -> int:
        return 0
    
    def discriminate(self, example: tuple) -> int:
        """Model prediction on a single adaptation task example.

        Args:
            example: a tuple representing the (tweet, rephrase) pair.
        
        Returns:
            an integer representing the index of the sarcastic tweet.
        """
        return self.classify()


class DeepMojiClassifier(Model):
    """Class that creates a model using DeepMoji pre-trained embeddings and predicts using a classifier trained on these embeddings.
        """

    def init_deepmoji(self, classifier, hyperparameters: dict):
        self.deepmoji = DeepMoji()
        self.classifier = classifier(**hyperparameters)

    def fit(self, train_examples, train_labels) -> None:
        """Trains a classifier on train data using vector embeddings obtained by feeding train data to the DeepMoji model.

        Args:
            train_examples: list of strings representing the text examples
            train_labels: list of integers that are the labels of the train examples, 1 for `sarcastic' and 0 otherwise.
        """

        # Get the global deepmoji train vectors instance
        vectors = self.deepmoji.get_train_vectors(train_examples)

        # fit the classifier to the embeddings and labels
        self.classifier.fit(vectors, train_labels)

    def predict(self, examples, split='', task="primary") -> list:
        """Get the model predictions on a list of text examples.

        Args:
            examples: list of strings representing the text examples
            split: string representing class of data (train, dev, test)

        Returns: 
            the list of predictions
        """
        prediction = self.classify

        if split == 'train':
            vectors = self.deepmoji.get_train_vectors(examples)
        elif split == 'dev':
            vectors = self.deepmoji.get_dev_vectors(examples)
        elif split == 'test':
            vectors = self.deepmoji.get_test_vectors(examples)
        elif not split and task == "adaptation":
            vectors = self.deepmoji.get_adaptation_vectors(examples)
            prediction = self.discriminate
        else:
            raise ValueError("split must be train, dev, or test but received: {0}".format(split))

        # return [self.classify(vector) for vector in vectors]
        return [prediction(vector) for vector in vectors]

    def classify(self, *args, **kwargs) -> int:
    # def classify(self, example, is_sentence=False) -> int:
        """
        Classifies example by using the example's DeepMoji vector as the input of classifier's predict function.
        Predicts false on empty strings to avoid breaking DeepMoji system.

        Args:
            example: a string of text

        Return:
            an integer representing the predicted label
        """
        example = args[0]
        # is_sentence = kwargs['is_sentence']

        # if is_sentence:
            # example = self.deepmoji.embedding(example)
        # example = self.deepmoji.embedding(example)
        predicted_label = self.classifier.predict([example])[0]
        return predicted_label

    def discriminate(self, example: tuple) -> int:
        """Model prediction on a single adaptation task example.

        Use the sklearn get_proba() function to get the trained classifier's sarcasm score for the pair of sentences, and return the index corresponding to the sentence with highest score.

        Args:
            example: a tuple representing the (tweet, rephrase) pair -- not necessarily in that order.
        
        Returns:
            an integer representing the index of the sarcastic tweet.
        """
        scores = distributions_over_labels(self.classifier, example)
        return np.argmax(scores[:,1])

class DeepMojiWithPolarityClassifier(DeepMojiClassifier):
    """DeepMoji classifier with polarity features appended to the end of each vector."""

    def fit(self, train_examples, train_labels) -> None:
        """Trains a classifier on train data using vector embeddings obtained by feeding train data to the DeepMoji model.

        Args:
            train_examples: list of strings representing the text examples
            train_labels: list of integers that are the labels of the train examples, 1 for `sarcastic' and 0 otherwise.
        """

        # Get the global deepmoji train vectors instance
        vectors = self.deepmoji.get_polarity_train_vectors(train_examples)

        # fit the classifier to the embeddings and labels
        self.classifier.fit(vectors, train_labels)

    def predict(self, examples, split='', task="primary") -> list:
        """Get the model predictions on a list of text examples.

        Args:
            examples: list of strings (primary) or list of pairs of strings (adaptation)

            is_train: boolean representing whether examples are from the training set

        Returns: 
            the list of predictions
        """
        prediction = self.classify
        if split == 'train':
            vectors = self.deepmoji.get_polarity_train_vectors(examples)
        elif split == 'dev':
            vectors = self.deepmoji.get_polarity_dev_vectors(examples)
        elif split == 'test':
            vectors = self.deepmoji.get_polarity_test_vectors(examples)
        elif not split and task == "adaptation":
            vectors = self.deepmoji.get_polarity_adaptation_vectors(examples)
            prediction = self.discriminate
        else:
            raise ValueError("split must be train, dev, or test but received: {0}".format(split))

        # return [self.classify(vector) for vector in vectors]
        return [prediction(vector) for vector in vectors]

class DeepMojiWithSVM(DeepMojiClassifier):
    def __init__(self, hyperparameters: dict) -> None:
        self.init_deepmoji(svm.SVC, hyperparameters)


class DeepMojiWithRandomForest(DeepMojiClassifier):
    def __init__(self, hyperparameters: dict) -> None:
        self.init_deepmoji(RandomForestClassifier, hyperparameters)


class DeepMojiWithAdaBoost(DeepMojiClassifier):
    def __init__(self, hyperparameters: dict) -> None:
        self.init_deepmoji(AdaBoostClassifier, hyperparameters)


class DeepMojiWithMLP(DeepMojiClassifier):
    def __init__(self, hyperparameters: dict) -> None:
        self.init_deepmoji(MLPClassifier, hyperparameters)


class DeepMojiWithNaiveBayes(DeepMojiClassifier):
    def __init__(self, hyperparameters: dict) -> None:
        self.init_deepmoji(GaussianNB, hyperparameters)


class DeepMojiWithKNN(DeepMojiClassifier):
    def __init__(self, hyperparameters: dict) -> None:
        self.init_deepmoji(KNeighborsClassifier, hyperparameters)


class DeepMojiWithLogisticRegression(DeepMojiClassifier):
    def __init__(self, hyperparameters: dict) -> None:
        self.init_deepmoji(LogisticRegression, hyperparameters)


class DeepMojiWithVoting(DeepMojiClassifier):
    def __init__(self, hyperparameters: dict) -> None:
        estimators = [('adaboost', AdaBoostClassifier()), ('mlp', MLPClassifier(alpha=1e-5, solver='adam')),
                      ('KNN', KNeighborsClassifier())]
        hyperparameters['estimators'] = estimators
        self.init_deepmoji(VotingClassifier, hyperparameters)


# Polarity-based DeepMoji classifier
class DeepMojiPolarityWithKNN(DeepMojiWithPolarityClassifier):
    def __init__(self, hyperparameters: dict) -> None:
        self.init_deepmoji(KNeighborsClassifier, hyperparameters)

class DeepMojiPolarityWithMLP(DeepMojiWithPolarityClassifier):
    def __init__(self, hyperparameters: dict) -> None:
        self.init_deepmoji(MLPClassifier, hyperparameters)

class DeepMojiPolarityWithAdaBoost(DeepMojiWithPolarityClassifier):
    def __init__(self, hyperparameters: dict) -> None:
        self.init_deepmoji(AdaBoostClassifier, hyperparameters)


class DeepMojiPolarityWithVoting(DeepMojiWithPolarityClassifier):
    def __init__(self, hyperparameters: dict) -> None:
        estimators = [('adaboost', AdaBoostClassifier()), ('mlp', MLPClassifier(alpha=1e-5, solver='adam')),
                      ('KNN', KNeighborsClassifier())]
        hyperparameters['estimators'] = estimators
        self.init_deepmoji(VotingClassifier, hyperparameters)
