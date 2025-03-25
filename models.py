import random
import numpy as np
import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        dp = self.run(x)
        scalar = nn.as_scalar(dp)
        if scalar >= 0:
            return 1
        else:
            return -1
        
    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        converged = False
        while not converged:
            converged = True
            for x, y in dataset.iterate_once(1):
                prediction = self.get_prediction(x)
                if prediction != nn.as_scalar(y):
                    converged = False
                    self.w.update(x, nn.as_scalar(y))
            
                    


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_layer_size = 512
        self.batch_size = 200
        self.learning_rate = -0.01
        self.init_weights = nn.Parameter(1, self.hidden_layer_size)
        self.res_weights = nn.Parameter(self.hidden_layer_size, 1)
        self.init_bias = nn.Parameter(1, self.hidden_layer_size)
        self.res_bias = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        #layer 1 
        init_lin = nn.Linear(x, self.init_weights)
        init_bias = nn.AddBias(init_lin, self.init_bias)
        init_relu = nn.ReLU(init_bias)
        #layer 2
        sec_lin = nn.Linear(init_relu, self.res_weights)
        sec_bias = nn.AddBias(sec_lin, self.res_bias)
        return sec_bias
        
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x,y)
                if nn.as_scalar(loss) <= 0.02:
                    return
                grad_IW, grad_IB, grad_RW, grad_RB = nn.gradients(loss, [self.init_weights, self.init_bias, self.res_weights, self.res_bias])
                self.init_weights.update(grad_IW, self.learning_rate)
                self.res_weights.update(grad_RW, self.learning_rate)
                self.init_bias.update(grad_IB, self.learning_rate)
                self.res_bias.update(grad_RB, self.learning_rate)
                


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.input_size = 784
        self.first_hidden_layer_size = 200
        self.second_hidden_layer_size = 70
        self.output_layer_size = 10
        self.batch_size = 100
        self.learning_rate = -0.2


        self.first_layer_weights = nn.Parameter(self.input_size, self.first_hidden_layer_size)
        self.second_layer_weights = nn.Parameter(self.first_hidden_layer_size, self.second_hidden_layer_size)
        self.output_weights = nn.Parameter(self.second_hidden_layer_size, self.output_layer_size)
        self.first_bias = nn.Parameter(1, self.first_hidden_layer_size)
        self.second_bias = nn.Parameter(1, self.second_hidden_layer_size)
        self.output_bias = nn.Parameter(1, self.output_layer_size)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        layer_1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.first_layer_weights), self.first_bias))
        layer_2 = nn.ReLU(nn.AddBias(nn.Linear(layer_1, self.second_layer_weights), self.second_bias))
        output_layer = nn.AddBias(nn.Linear(layer_2, self.output_weights), self.output_bias)
        return output_layer

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                validation = dataset.get_validation_accuracy()
                if validation >= 0.98:
                    return
                loss = self.get_loss(x, y)
                grad_fl, grad_sl, grad_ol, grad_fb, grad_sb, grad_ob = nn.gradients(loss, [self.first_layer_weights, self.second_layer_weights, self.output_weights, self.first_bias, self.second_bias, self.output_bias])
                self.first_layer_weights.update(grad_fl, self.learning_rate)
                self.second_layer_weights.update(grad_sl, self.learning_rate)
                self.output_weights.update(grad_ol, self.learning_rate)
                self.first_bias.update(grad_fb, self.learning_rate)
                self.second_bias.update(grad_sb, self.learning_rate)
                self.output_bias.update(grad_ob, self.learning_rate)

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.output_size = 5
        self.hidden_layer_size = 500
        self.batch_size = 200
        self.learning_rate = -0.5
        # self.init_bias = nn.Parameter(1, self.num_chars)
        self.hidden_bias = nn.Parameter(1, self.hidden_layer_size)
        self.output_bias = nn.Parameter(1, self.output_size)
        self.initial_weights = nn.Parameter(self.num_chars, self.hidden_layer_size)
        self.hidden_layer_weights = nn.Parameter(self.hidden_layer_size, self.hidden_layer_size)
        self.output_weights = nn.Parameter(self.hidden_layer_size, self.output_size)
       
        
    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        initial = nn.ReLU(nn.Linear(xs[0], self.initial_weights))
        length = len(xs)
        for i in range(1, length):
            initial = nn.ReLU(nn.AddBias(nn.Add(nn.Linear(xs[i], self.initial_weights), nn.Linear(initial, self.hidden_layer_weights)), self.hidden_bias))
        out = nn.AddBias(nn.Linear(initial, self.output_weights), self.output_bias)
        return out


    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                validation = dataset.get_validation_accuracy()
                if validation >= 0.85:
                    return
                loss = self.get_loss(x, y)
                grad_iW, grad_hW, grad_oW, grad_hB, grad_oB= nn.gradients(loss, [self.initial_weights, self.hidden_layer_weights, self.output_weights, self.hidden_bias, self.output_bias])
                self.initial_weights.update(grad_iW, self.learning_rate)
                self.hidden_layer_weights.update(grad_hW, self.learning_rate)
                self.output_weights.update(grad_oW, self.learning_rate)
                #self.init_bias.update(grad_iB, self.learning_rate)
                self.hidden_bias.update(grad_hB, self.learning_rate)
                self.output_bias.update(grad_oB, self.learning_rate)
                