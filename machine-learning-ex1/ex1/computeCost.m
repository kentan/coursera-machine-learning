function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.



iterations = 1500;
alpha = 0.01;


%for j = 1:iterations
%  for i = 1:m 
%	  h = theta'*[X(i),1]';
%	  theta(1) = theta(1) - alpha * (1/m) * (h - y(i));
%	  theta(2) = theta(2) - alpha * (1/m) * (h - y(i))*X(i);
%  endfor
%endfor


sum = 0;
for i = 1:m 
  h = theta' * [1,X(i,2)]';
  sum = sum +  (( h - y(i)).^2);
  

endfor


J =  sum/2/m;

% =========================================================================

end
