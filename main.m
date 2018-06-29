
clear ; close all; clc

ctg_data = dlmread('catprocd-bank-additional-full.csv',',' ,1,0);
X = dlmread('numprocd-bank-additional-full.csv',',' ,1,0);

[x_cat, y_cat] = size(ctg_data);
y = ctg_data(:, y_cat);

%%%%Convert categorical features to binary features

for i=1:(y_cat-1)
    X = [X createBinFeatures(ctg_data(:,i))];
end
X = featureNormalize(X);

[m, n] = size(X);


%%%%%%% Split data into training set, cross validation set and test set

XTrain = X(1:25000,:);
yTrain = y(1:25000,:);
mTrain = size(XTrain, 1);

XCV = X(25001:33000, :);
yCV = y(25001: 33000,:);
mCV = size(XCV, 1);

XTest = X(33001:m, :);
yTest = y(33001:m, :);
mTest = size(XTest, 1);




%%%%%% Run logistic regression on training set %%%%%%%%%%%%




%%%%%%%%     Plot learning curves         %%%%%%%%%%%%

% lambda = 1 ;
% INTERVAL = 500;
% num_iter = floor(size(XTrain, 1)/INTERVAL);

% [error_train, error_val] = ...
%     learningCurve([ones(mTrain, 1) XTrain], yTrain, ...
%                   [ones(size(XCV, 1), 1) XCV], yCV, ...
%                   lambda, INTERVAL );





%%%%   Try out different values of d and lambda and pick the one with best CV performance

% try our dimensions 1 to 6 
% Try to 10 powers for lambda
best_lambda = -1;
best_d = -1;
best_theta = [];
min_cv_error = 100;

for d=1:1
    fprintf("\nd= %d || ", d);
    
    XPolyTrain = [ones(mTrain, 1)];
    XPolyCV = [ones(mCV, 1)];
    for i=1:n
        XPolyTrain = [XPolyTrain polyFeatures(XTrain(:,i), d)];
        XPolyCV = [XPolyCV polyFeatures(XCV(:,i), d)];
    end 
    fprintf("Cols in PolyTrain: %d, Cols in PolyCv: %d ", size(XPolyTrain, 2), size(XPolyCV, 2));


    for p=1:14
        fprintf("p=%d, ", p);


        theta = trainLogisticReg(XPolyTrain, yTrain, 0.01*(2^p));
        cv_error = 100- mean(double(predict(theta, XPolyCV) == yCV)) * 100;
        if ( cv_error < min_cv_error)
            best_d = d;
            best_lambda = 0.01*(2^p);
            fprintf("Best d so far: %d, Best lambda so far %f, CV error %f\n", best_d, best_lambda, cv_error);
            best_theta = theta;
            min_cv_error = cv_error;
            fprintf("Theta size: %d\n", size(theta));
        endif
    end
end




fprintf('Best values of d and lambda are %d , %f\n', best_d, best_lambda);

% Calc Test set error
XTestPoly = [ones(mTest, 1)];
for i=1:n
    XTestPoly = [XTestPoly polyFeatures(XTest(:,i), best_d)];
end
p = predict(theta, XTestPoly);

fprintf('Test Error: %f\n', 100 - mean(double(p == yTest)) * 100);
fprintf("Theta size: %d\n", size(theta, 1));
fprintf("XPolyTest size: %d\n", size(XTestPoly));
fprintf("n: %d\n", n);






% [theta, lambda] = getBestParams(XTrain, yTrain, XCV, yCV, initial_theta);


% %%%%%% Choose value of parameters (lambda: any more?) on cross validation set



% %%%%%% Test values on test set and visualize results. Get Performance 


% p = predict(theta, XTest);

% fprintf('Train Accuracy: %f\n', mean(double(p == yTest)) * 100);



