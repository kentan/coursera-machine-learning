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
a1 = [ones(m,1),X];
z2 = Theta1 * a1';
a2 = [ones(1,m);sigmoid(z2)];
z3 = Theta2 * a2;
a3 = sigmoid(z3);

y_ = zeros(num_labels,m);
for i=1:m
  for j=1:num_labels
     c = (y(i) == j);
     y_(j,i) = c;
  end
end
y;
y_;

s = 0;
for j=1:m
   for i=1:num_labels
       c = (y(j) == i);
       s += -c*log(a3(i,j))-(1-c)*log((1-a3)(i,j));
   end
end

J = s/m;

t1 = [zeros(size(Theta1,1)),Theta1(:,2:end)];
t2 = [zeros(size(Theta2,1)),Theta2(:,2:end)];

J += (lambda * (sum(sum(t1 .^2)) + sum(sum(t2 .^ 2))))/(2*m);

d3 = a3 - y_;
d2 = (Theta2(:,2:end)' * d3) .* sigmoidGradient(z2);
%d2 = d2(2:end,:);
%size(a3);
%size(y_);
%size(Theta1);
%size(Theta2);
%size(z2);
%size(a2);
%size(d2);

wd2 = 0;
wd1 = 0;


for i=1:m
   wd2 += d3(:,i) * a2(:,i)';
   wd1 += d2(:,i) * a1(i,:);
end
Theta2_grad = wd2/m;
Theta1_grad =  wd1/m;
%size(Theta1_grad);
%size(Theta1_grad2);

Theta1_grad(:,2:end) += Theta1(:,2:end)*(lambda/m);
Theta2_grad(:,2:end) += Theta2(:,2:end)*(lambda/m);
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
