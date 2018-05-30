function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% call nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

% forward propagation implementation
a1 = [ones(m,1) X];

z2 = a1 * Theta1';
a2 = [ones(m,1) sigmoid(z2)];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

% formatting the output into vector.
actual = zeros(m, num_labels);
for i = 1:m
  actual(i, y(i)) = 1;
end
% size(actual)


% computing cost

log_h = log(a3);
log_1_h = log(1 .- a3);

% cost function without regularization
J = (1/m) * -sum(sum((actual .* log_h) + ((1 .- actual) .* log_1_h )));

% regularization in cost function

Theta1_no_bias = Theta1(:, 2:end);
Theta2_no_bias = Theta2(:, 2:end);

J += (lambda/ (2*m)) * (sum(sum(Theta1_no_bias .^ 2)) + sum(sum(Theta2_no_bias .^ 2)));


% implementing the backpropagation with for loop.

for i = 1:m
  a1_temp = a1(i, :);
  z2_temp = z2(i, :);
  a2_temp = a2(i, :);
  z3_temp = z3(i, :);
  a3_temp = a3(i, :);

  % calculating delta (difference between prediction and actual values) without regularization
  delta3 = a3_temp - actual(i, :);
  delta2 = delta3 * Theta2_no_bias .* sigmoidGradient(z2_temp);
  Theta2_grad += delta3' * a2_temp;
  Theta1_grad += delta2' * a1_temp;
end


Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;

% adding regularization to gradients

Theta1_grad(:, 2:end) += (lambda / m) .* Theta1_no_bias;
Theta2_grad(:, 2:end) += (lambda / m) .* Theta2_no_bias;

% Theta1_grad += ;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
