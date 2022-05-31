%Feed-Forward Classification Network (Commands)
%Instructions are in the task pane to the left. Complete and submit each task one at a time.
%This code loads and displays the heart disease data set.
load heartDisease
whos HDC heartData

%Task 1
%Initialize a classification neural network named net with one hidden layer containing 15 neurons.
%Divide the data so that 70% is for training, 10% is for testing, and 20% is for validation.
%Initialize neural network
net = patternnet(15);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 10/100;

%Task 2
%Train the network net with the data in heartData and the target values in HDC. Save the training record to tr.

%Note that the target values are stored as categorical data, and samples for the data should be along the matrix columns rather than rows.
%Train the network
heartData = heartData';  % transpose samples
HD = dummyvar(HDC)';     % convert categorical to dummy variable
[net,tr] = train(net,heartData,HD);

%Task 3
%Make predictions on the test data. Store the predicted values in a variable named scoreTest.
%Then, create a vector named yPred which contains the row index value with the largest value for each column in the predicted matrix.
%Predict response
scoreTest = net(heartData(:,tr.testInd));
[~,yPred] = max(scoreTest);

%Task 4
%Evaluate classification with confusion matrix
HDtest = HDC(tr.testInd);
yTrue = double(HDtest);
confusionchart(yTrue,yPred');

%Task 5
% Calculate the percentage of misclassified predicted test values and save the value in a variable named validErr.
%Determine validation error
HDtest = HDC(tr.testInd);
validErr = 100*nnz(yPred' ~= double(HDtest))/length(HDtest);
disp(validErr)
