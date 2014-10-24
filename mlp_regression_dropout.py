import numpy as np
import cPickle
import gzip
import os
import sys
import time
from collections import OrderedDict

import warnings
warnings.filterwarnings('ignore')

import theano
import theano.tensor as T
from theano.ifelse import ifelse
import theano.printing
import theano.tensor.shared_randomstreams

from load_data import load_data


##################################
## Various activation functions ##
##################################
#### rectified linear unit
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
#### sigmoid
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
#### tanh
def Tanh(x):
    y = T.tanh(x)
    return(y)
    
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,
                 activation, W=None, b=None,
                 use_bias=False):

        self.input = input
        self.activation = activation

        if W is None:
            W_values = np.asarray(0.01 * rng.standard_normal(
                size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W')
        
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        if use_bias:
            lin_output = T.dot(input, self.W) + self.b
        else:
            lin_output = T.dot(input, self.W)

        self.output = (lin_output if activation is None else activation(lin_output))
    
        # parameters of the model
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]
            
class Regression(object):
    def __init__(self, input, n_in, n_out, W=None, b=None):

        if W is None:
            W = theano.shared(value=np.zeros((n_in, n_out),
                               dtype=theano.config.floatX),
                               name='W', borrow=True)
        
        if b is None:
            b = theano.shared(value=np.zeros((n_out,),
                               dtype=theano.config.floatX),
                               name='b', borrow=True)

        self.W = W
        self.b = b

        # compute vector of real values in symbolic form
        self.y_pred = T.dot(input, self.W) + self.b

        # parameters of the model
        self.params = [self.W, self.b]
        
    def predict(self):
        return self.y_pred

    def errors(self, y):

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        else:
            return T.sqrt(T.mean((self.y_pred - y)**2, axis = 0)) #returns the RMSE for each category
            
    def MCRMSE(self, y):
        return T.mean(T.sqrt(T.mean((self.y_pred - y)**2, axis = 0))) #returns the average RMSE of the five categories, or the MCRMSE
        
    def MSE(self, y):
        return T.mean((self.y_pred - y)**2)

def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, dropout_rate, use_bias, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation, use_bias=use_bias)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)


class MLP(object):
    """A multilayer perceptron with all the trappings required to do dropout
    training.

    """
    def __init__(self,
            rng,
            input,
            layer_sizes,
            dropout_rates,
            activations,
            use_bias=True):

        #rectified_linear_activation = lambda x: T.maximum(0.0, x)

        # Set up all the hidden layers
        weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        self.dropout_layers = []
        next_layer_input = input
        #first_layer = True
        # dropout the input
        next_dropout_layer_input = _dropout_from_layer(rng, input, p=dropout_rates[0])
        layer_counter = 0        
        for n_in, n_out in weight_matrix_sizes[:-1]:
            next_dropout_layer = DropoutHiddenLayer(rng=rng,
                    input=next_dropout_layer_input,
                    activation=activations[layer_counter],
                    n_in=n_in, n_out=n_out, use_bias=use_bias,
                    dropout_rate=dropout_rates[layer_counter + 1])
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output

            # Reuse the paramters from the dropout layer here, in a different
            # path through the graph.
            next_layer = HiddenLayer(rng=rng,
                    input=next_layer_input,
                    activation=activations[layer_counter],
                    # scale the weight matrix W with (1-p)
                    W=next_dropout_layer.W * (1 - dropout_rates[layer_counter]),
                    b=next_dropout_layer.b,
                    n_in=n_in, n_out=n_out,
                    use_bias=use_bias)
            self.layers.append(next_layer)
            next_layer_input = next_layer.output
            #first_layer = False
            layer_counter += 1
        
        # Set up the output layer
        n_in, n_out = weight_matrix_sizes[-1]
        dropout_output_layer = Regression(
                input=next_dropout_layer_input,
                n_in=n_in, n_out=n_out)
        self.dropout_layers.append(dropout_output_layer)

        # Again, reuse paramters in the dropout output.
        output_layer = Regression(
            input=next_layer_input,
            # scale the weight matrix W with (1-p)
            W=dropout_output_layer.W * (1 - dropout_rates[-1]),
            b=dropout_output_layer.b,
            n_in=n_in, n_out=n_out)
        self.layers.append(output_layer)

        # Use the negative log likelihood of the logistic regression layer as
        # the objective.
        self.dropout_negative_log_likelihood = self.dropout_layers[-1].MCRMSE #cost function
        self.dropout_errors = self.dropout_layers[-1].errors #error function

        self.negative_log_likelihood = self.layers[-1].MCRMSE #cost function
        self.errors = self.layers[-1].errors #error function
        
        self.predict = self.layers[-1].predict

        # Grab all the parameters together.
        self.params = [ param for layer in self.dropout_layers for param in layer.params ]


def test_mlp(
        initial_learning_rate,
        learning_rate_decay,
        squared_filter_length_limit,
        n_epochs,
        batch_size,
        mom_params,
        activations,
        dropout,
        dropout_rates,
        results_file_name,
        layer_sizes,
        datasets,
        Z,
        use_bias,
        fold,
        random_seed=1234):
    """
    The dataset is the one from the mlp demo on deeplearning.net.  This training
    function is lifted from there almost exactly.

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


    """
    assert len(layer_sizes) - 1 == len(dropout_rates)
    
    # extract the params for momentum
    mom_start = mom_params["start"]
    mom_end = mom_params["end"]
    mom_epoch_interval = mom_params["interval"]
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    epoch = T.scalar()
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.matrix('y')  # the labels are presented as 1D vector of
                        # [int] labels
    learning_rate = theano.shared(np.asarray(initial_learning_rate,
        dtype=theano.config.floatX))

    rng = np.random.RandomState(random_seed)

    # construct the MLP class
    classifier = MLP(rng=rng, input=x,
                     layer_sizes=layer_sizes,
                     dropout_rates=dropout_rates,
                     activations=activations,
                     use_bias=use_bias)

    # Build the expresson for the cost function.
    cost = classifier.negative_log_likelihood(y)
    dropout_cost = classifier.dropout_negative_log_likelihood(y)
 
    # Compile theano function for testing.
    test_model = theano.function(inputs = [],
            outputs=classifier.errors(y),
            givens={x: test_set_x, y: test_set_y})
    #theano.printing.pydotprint(test_model, outfile="test_file.png",
    #        var_with_name_simple=True)
    
    
    # Compile theano function for validation.
    validate_model = theano.function(inputs=[],
            outputs=classifier.errors(y),
            givens={x: valid_set_x, y: valid_set_y})
    #theano.printing.pydotprint(validate_model, outfile="validate_file.png",
    #        var_with_name_simple=True)

    # Compute gradients of the model wrt parameters
    gparams = []
    for param in classifier.params:
        # Use the right cost function here to train with or without dropout.
        gparam = T.grad(dropout_cost if dropout else cost, param)
        gparams.append(gparam)

    # ... and allocate mmeory for momentum'd versions of the gradient
    gparams_mom = []
    for param in classifier.params:
        gparam_mom = theano.shared(np.zeros(param.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
        gparams_mom.append(gparam_mom)

    # Compute momentum for the current epoch
    mom = ifelse(epoch < mom_epoch_interval,
            mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval),
            mom_end)

    # Update the step direction using momentum
    updates = OrderedDict()
    for gparam_mom, gparam in zip(gparams_mom, gparams):
        # Misha Denil's original version
        #updates[gparam_mom] = mom * gparam_mom + (1. - mom) * gparam
      
        # change the update rule to match Hinton's dropout paper
        updates[gparam_mom] = mom * gparam_mom - (1. - mom) * learning_rate * gparam

    # ... and take a step along that direction
    for param, gparam_mom in zip(classifier.params, gparams_mom):
        # Misha Denil's original version
        #stepped_param = param - learning_rate * updates[gparam_mom]
        
        # since we have included learning_rate in gparam_mom, we don't need it
        # here
        stepped_param = param + updates[gparam_mom]

        # This is a silly hack to constrain the norms of the rows of the weight
        # matrices.  This just checks if there are two dimensions to the
        # parameter and constrains it if so... maybe this is a bit silly but it
        # should work for now.
        if param.get_value(borrow=True).ndim == 2:
            #squared_norms = T.sum(stepped_param**2, axis=1).reshape((stepped_param.shape[0],1))
            #scale = T.clip(T.sqrt(squared_filter_length_limit / squared_norms), 0., 1.)
            #updates[param] = stepped_param * scale
            
            # constrain the norms of the COLUMNs of the weight, according to
            # https://github.com/BVLC/caffe/issues/109
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(squared_filter_length_limit))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param


    # Compile theano function for training.  This returns the training cost and
    # updates the model parameters.
    output = dropout_cost if dropout else cost
    train_model = theano.function(inputs=[epoch, index], outputs=output,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})
    #theano.printing.pydotprint(train_model, outfile="train_file.png",
    #        var_with_name_simple=True)

    # Theano function to decay the learning rate, this is separate from the
    # training function because we only want to do this once each epoch instead
    # of after each minibatch.
    
    predict = theano.function(inputs = [], outputs=classifier.predict(), givens={x: Z})
    predict_valid = theano.function(inputs = [], outputs=classifier.predict(), givens={x: valid_set_x})
    
    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
            updates={learning_rate: learning_rate * learning_rate_decay})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    best_params = None
    best_validation_errors = np.inf
    best_iter = 0
    test_score = 0.
    epoch_counter = 0
    start_time = time.clock()

    results_file = open(results_file_name, 'wb')
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    while epoch_counter < n_epochs:
        # Train this epoch
        epoch_counter = epoch_counter + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(epoch_counter, minibatch_index)
            
        # Compute loss on validation set
        validation_losses = validate_model()
        this_validation_errors = validation_losses

        # Report and save progress.
        print "epoch {}, RMSE {} MCRMSE {}, lr={}{}".format(
                epoch_counter, this_validation_errors, round(np.mean(this_validation_errors),3),
                round(learning_rate.get_value(borrow=True),4),
                " **" if np.mean(this_validation_errors) < best_validation_errors else "")

        new_learning_rate = decay_learning_rate()
        best_validation_errors = min(best_validation_errors, np.mean(this_validation_errors))

    prediction = predict()
    np.savetxt("soil_predictions" + str(fold) + ".csv", prediction, delimiter=",")
    return prediction, np.mean(this_validation_errors)

if __name__ == '__main__':

    import sys
    
    prediction = []
    errors = []
    
    for fold in range(50):
        print "Running iter " + str(fold)
        random_seed = fold#1234
        dataset, Z = load_data(path = '../training.csv', fold = fold)
        n_input = dataset[0][0].get_value(borrow=True).shape[1]
        n_output = dataset[0][1].get_value(borrow=True).shape[1]
        
        initial_learning_rate = 0.1
        learning_rate_decay = 0.998
        squared_filter_length_limit = 15.0
        n_epochs = 1050
        batch_size = 100
        layer_sizes = [ n_input, 250, 250, 100, n_output ] #inputs, h1, h2, h3, outputs
        
        # dropout rate for each layer
        dropout_rates = [ 0, 0.2, 0.2, 0 ]
        # activation functions for each layer
        # For this demo, we don't need to set the activation functions for the 
        # on top layer, since it is always 10-way Softmax
        activations = [ ReLU, ReLU, None ]
        
        #### the params for momentum
        mom_start = 0.5
        mom_end = 0.99
        # for epoch in [0, mom_epoch_interval], the momentum increases linearly
        # from mom_start to mom_end. After mom_epoch_interval, it stay at mom_end
        mom_epoch_interval = 500
        mom_params = {"start": mom_start,
                      "end": mom_end,
                      "interval": mom_epoch_interval}
                      
        dropout = True
        results_file_name = "results_backprop.txt"

        this_prediction, this_errors = test_mlp(initial_learning_rate=initial_learning_rate,
                 learning_rate_decay=learning_rate_decay,
                 squared_filter_length_limit=squared_filter_length_limit,
                 n_epochs=n_epochs,
                 batch_size=batch_size,
                 layer_sizes=layer_sizes,
                 mom_params=mom_params,
                 activations=activations,
                 dropout=dropout,
                 dropout_rates=dropout_rates,
                 datasets=dataset,
                 Z = Z,
                 results_file_name=results_file_name,
                 use_bias=False,
                 fold = fold,
                 random_seed=random_seed)
        prediction.append(this_prediction)
        errors.append(this_errors)
    errors = np.asarray(errors)
    model_average = np.mean(errors)
    
    prediction = [prediction[x] for x in range(len(errors)) if errors[x] < model_average]          
    prediction = np.mean(np.asarray(prediction), axis = 0)
    np.savetxt("soil_prediction_ensemble.csv", prediction, delimiter = ",")

                 


