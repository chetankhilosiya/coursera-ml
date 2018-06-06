
clear;


load('ANDdata1.mat');
load('bitwiseWeights.mat');

nn_params = Theta1(:);

input_layer_size = 2;
hidden_layer_size = 1;
num_labels = 1;

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_nn_params = initial_Theta1(:);

lambda = 0;


options = optimset('MaxIter', 30);

costFunction = @(p) bitwiseCost(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% predicting the output from input and theta

m = size(X, 1);

p = zeros(m, 1);

h1 = sigmoid([ones(m, 1) X] *  Theta1');
p = h1 > 0.5;


fprintf('\nTraining Set Accuracy: %f\n', mean(double(p == y)) * 100);
