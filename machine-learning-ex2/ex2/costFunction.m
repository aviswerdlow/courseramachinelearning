function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


h = sigmoid(X*theta); 

% ================= DEBUGING ===============================

%disp(size(h));

%disp(size(-y'*log(h')));

%disp(size(-y)');

%disp(size(log(h)));

%disp(size(log(1-(h)))); 

%disp(size((1-y)'));

%disp(size(m));

% ==========================================================

% Vectorized Implementation of Cost Function (J)
J = (1/m) * ((-y'*log(h)) - (1-y)' * log(1-h));

% Vectorized Implementation of Gradient Descent (grad)
grad = (X'*(h-y))/m;

% Reference site for both: https://www.coursera.org/learn/machine-learning/supplement/0hpMl/simplified-cost-function-and-gradient-descent

% =============================================================

end
