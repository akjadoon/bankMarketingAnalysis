function [error_train, error_val]= ...
    learningCurve(X, y, Xval, yval, lambda, INTERVAL)


% Number of iterations for learning curve
num_iter = floor(size(X, 1)/INTERVAL);

% You need to return these values correctly
error_train = zeros(num_iter, 1);
error_val   = zeros(num_iter, 1);


for i = 1:num_iter
    theta = trainLogisticReg(X(1:i*INTERVAL, :), y(1:i*INTERVAL), lambda);
    error_train(i) = 100- mean(double(predict(theta, X(1:i*INTERVAL, :)) == y(1:i*INTERVAL, :))) * 100;
    error_val(i) = 100- mean(double(predict(theta, Xval) == yval)) * 100;
end

plot(1:num_iter, error_train, 1:num_iter, error_val);
title('Learning curve for Logistic regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:num_iter
    fprintf('  \t%d\t\t%f\t%f\n', i*500, error_train(i), error_val(i));
end