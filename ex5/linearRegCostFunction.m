function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J1 = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
hypo = X * theta;
for i = 1:m
    J1 = J1 + (hypo(i) - y(i))^2;
end
J1 = J1/2/m;

J = J1 + lambda / 2 / m * (theta(2:end)' * theta(2:end));

n = length(theta);
for j = 1:n
    if j == 1
        grad(j) = 1/m * (hypo - y)' * X(:, 1);
    else
        grad(j) = 1/m * (hypo - y)' * X(:, j) + lambda / m * theta(j);
    end
    
end

% =========================================================================

grad = grad(:);

end
