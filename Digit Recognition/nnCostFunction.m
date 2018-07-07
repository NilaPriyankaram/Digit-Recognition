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



%Part 1 - Feed Forward Propgation
% Theta values used are Preloaded. Those values may be random


X = [ones(m,1), X] ;	%Adding ones in front of X. i.e) Adding bias unit. m is size(X,1) . i.e) No of training samples

z2 = (X * Theta1') ;	% a_2 is 5000 x 25

a_2 = sigmoid(z2);

a_2 = [ones(m,1), a_2] ;	%Adding ones a_2 is 5000 x 26

z3 = (a_2 * Theta2');	% a_3 is 5000 x 10

a_3 = sigmoid(z3) ;

y_matrix = eye(num_labels)(y,:) ;	%Converting y into it's binary equivalent. Use dummyvar(y) for Matlab. or bsxfun(@eq, y, [1:max(y)])

J = -sum(sum( y_matrix.*log(a_3) + (1-y_matrix).*log(1-a_3) ))/m ;	% y_matrix is also 5000 x 10. Element wise multiplication makes sense

#J = (-sum(sum(y_matrix.*log(a_3))) - sum(sum((1-y_matrix).*(log(1-a_3)))))/m; This can also be used

#J = -trace((y_matrix*log(a_3')) + (1-y_matrix)*(log(1-a_3')))/(m) ;  %This also works. Trace computes sum of diagonals

%Adding Regularization

J = J + (lambda / (2*m)) * ( sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) );

% Theta(:, 2:end) gives all values of Theta except first column. Bias unit values - So no need to regularise



%-------------Part 2 - Back Propagation----------------%


del_3 = a_3 - y_matrix ;	% Both are 5000 x 10. Each row representing. Has same dimension as a_3

gdash_2 = sigmoidGradient(z2) ;

del_2 = (del_3 * Theta2) .* [ones(size(z2,1),1), gdash_2] ;

del_2 = del_2(:, 2:end) ;	% Removing first element i.e) Bias unit


D2 = del_3' * a_2 ;	% Del(l) = Del(l) + del(l+1) a(l) for all l. So on Vectorising we get this.

D1 = del_2' * X ;	% D has same dimensions as Theta. a_1 is X with bias


%------------ Part 3 - Computing Gradient-------------%

Theta1_grad = (1/m) * D1 ;

Theta2_grad = (1/m) * D2 ;	% Dimension 10 x 26. This is actually average gradient for all samples.

%------------ Part 4 - Regularisation of the Gradient----------------%

Theta1_grad(:, 2:end) = Theta1_grad(:,2:end) + (lambda/m) * (Theta1(:,2:end)) ;
Theta2_grad(:, 2:end) = Theta2_grad(:,2:end) + (lambda/m) * (Theta2(:,2:end)) ; %First column ignored coz bias unit shouldn't be considered in ...
										%regularization

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
