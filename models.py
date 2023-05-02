import nn

class PerceptronModel(object):
    def __init__(self, dim):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dim` is the dimensionality of the data.
        For example, dim=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dim)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x_point):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x_point: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        weights = self.get_weights()
        # dot = 0
        # for weight in weights:
        #     ab = weight * x_point
        #     dot += ab
        return nn.DotProduct(weights, x_point)

    def get_prediction(self, x_point):
        """
        Calculates the predicted class for a single data point `x_point`.

        Returns: -1 or 1
        """
        dot = self.run(x_point)
        scalar = nn.as_scalar(dot)
        if scalar >= 0:
            return 1
        else:
            return -1
    def train_model(self, dataset):
        """
        Train the perceptron until convergence.
        """
        set_complete = False
        while not set_complete:
            num_data = 0 # number of elements in the set
            correct_data = 0 # number that are correct

            for x,y in dataset.iterate_once(1):
                # take each element and get the predicted value
                num_data += 1
                predicted_y = self.get_prediction(x)
                real_y = nn.as_scalar(y)
                # if the predicted value is correct, update the # of correct elems
                if predicted_y == real_y:
                    correct_data += 1
                # if the predicted value is wrong, update the weights
                else:
                    self.w.update(real_y, x)
            # check if all elements are correct
            if(num_data == correct_data):
                set_complete = True

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        self.limit = 0.02 # loss limit (given)

        # f(x) = relu(x * W1 + b1) * W2 + b2 (from NN Tips)
        # need vars w1, w2, b1, b2
        # the comments were taken from the assignment sheet:
        
        # We are free to choose any value we want for the hidden size
        self.h = 400 # between 1-400

        # W1 will be an i × h matrix, where i s the dimension of our 
        # input vectors x , and h is the hidden layer size.
        self.w1 = nn.Parameter(1, self.h)
        # b1 will be a size h vector
        self.b1 = nn.Parameter(1, self.h)

        self.w2 = nn.Parameter(self.h, 1) # h x output size
        self.b2 = nn.Parameter(1,1) # 1 x output size

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        # f(x) = relu(x * W1 + b1) * W2 + b2 (from NN Tips)
        xw1 = nn.Linear(x, self.w1) # x * W1
        prediction = nn.AddBias(xw1, self.b1) # x * W1 + b1
        relu = nn.ReLU(prediction) # relu(x * W1 + b1)
        reluw2 = nn.Linear(relu, self.w2) # relu(x * W1 + b1) * W2
        f = nn.AddBias(reluw2, self.b2) # relu(x * W1 + b1) * W2 + b2

        return f

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        prediction = self.run(x)
        return nn.SquareLoss(prediction, y)
    def train_model(self, dataset):
        """
        Trains the model.
        """
        loss002 = False # from assignment description
        while not loss002:
            total_loss = 0
            num_data = 0
            # batch size: 1
            for x,y in dataset.iterate_once(1):
                num_data += 1

                loss = self.get_loss(x,y)
                total_loss += nn.as_scalar(loss)

                # from assignment example:
                # grad_wrt_m, grad_wrt_b = nn.gradients([m, b], loss)
                grad_wrt_w1, grad_wrt_w2, grad_wrt_b1, grad_wrt_b2 = nn.gradients([self.w1, self.w2, self.b1, self.b2], loss)

                # learning rate: 0.001
                self.w1.update(-0.001, grad_wrt_w1)
                self.w2.update(-0.001, grad_wrt_w2)
                self.b1.update(-0.001, grad_wrt_b1)
                self.b2.update(-0.001, grad_wrt_b2)

            if (total_loss/num_data < self.limit):
                loss002 = True

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).
    """
    def __init__(self):
        self.h = 400 # between 1-400
        self.h2 = 200 # between 1-400
        # f(x) = relu( relu(x * W1 + b1) * W2 + b2) * W3 + b3
        self.w1 = nn.Parameter(784, self.h) # input size x h1
        self.b1 = nn.Parameter(1, self.h) # 1 x h1
        self.w2 = nn.Parameter(self.h, self.h2) # h1 x h2
        self.b2 = nn.Parameter(1, self.h2) # 1 x h2
        self.w3 = nn.Parameter(self.h2, 10) # h2 x output size
        self.b3 = nn.Parameter(1, 10) # 1 x output size

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        # f(x) = relu( relu(x * W1 + b1) * W2 + b2) * W3 + b3
        xw1 = nn.Linear(x, self.w1) # x * W1
        prediction = nn.AddBias(xw1, self.b1) # x * W1 + b1
        relu = nn.ReLU(prediction) # relu(x * W1 + b1)
        reluw2 = nn.Linear(relu, self.w2) # relu(x * W1 + b1) * W2
        prediction2 = nn.AddBias(reluw2, self.b2) # relu(x * W1 + b1) * W2 + b2
        relu2 = nn.ReLU(prediction2) # relu( relu(x * W1 + b1) * W2 + b2)
        reluw3 = nn.Linear(relu2, self.w3) # relu( relu(x * W1 + b1) * W2 + b2) * W3
        f = nn.AddBias(reluw3, self.b3) # relu( relu(x * W1 + b1) * W2 + b2) * W3 + b3

        return f

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
        prediction = self.run(x)
        return nn.SoftmaxLoss(prediction, y)

    def train_model(self, dataset):
        """
        Trains the model.
        """
        accuracy = 0
        while accuracy < 0.975: # from assignment sheet
            # we require that total size of the dataset be evenly divisible by the batch size
            for x,y in dataset.iterate_once(75): # 60000/75=800
                loss = self.get_loss(x,y)
                grad_wrt_w1, grad_wrt_w2, grad_wrt_w3, grad_wrt_b1, grad_wrt_b2, grad_wrt_b3 = nn.gradients([self.w1, self.w2, self.w3, self.b1, self.b2, self.b3], loss)
                # learning rate: 0.1
                self.w1.update(-0.1, grad_wrt_w1)
                self.w2.update(-0.1, grad_wrt_w2)
                self.w3.update(-0.1, grad_wrt_w3)
                self.b1.update(-0.1, grad_wrt_b1)
                self.b2.update(-0.1, grad_wrt_b2)
                self.b3.update(-0.1, grad_wrt_b3)

            accuracy = dataset.get_validation_accuracy()
