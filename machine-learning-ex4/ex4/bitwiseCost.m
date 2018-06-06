function [J grad] = bitwiseCost(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));


% implementation

sleep(0.1);

a1 = [ones(m, 1) X];

z2 = a1 * Theta1';
a2 = sigmoid(z2);

log_h = log(a2);
log_1_h = log(1 .- a2);

J = (1/m) .* -sum(sum((y .* log_h) + ((1 .- y) .* log_1_h)));

for i = 1:m
  z2_t = z2(i, :);
  a2_t = a2(i, :);
  y_t = y(i, :);
  a1_t = a1(i, 2:end);

  delta2 = a2_t - y_t;
  Theta1_grad += delta2' .* a1_t(:, 2:end);
end

Theta1_grad = Theta1_grad ./ m;


grad= Theta1_grad(:);

end
