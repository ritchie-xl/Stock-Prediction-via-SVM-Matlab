clear;
clc;

%load financial data of the stock price of Apple company
%The data is from Nov 18 1982-Nov 18 2012
%The data contains six collums:Open, High, Low, Close, Volume, Adj Close
sh = dlmread('yahoo.csv');
%The data needs to flip because the data is from latest to earliest.
sh = flipdim(sh,1);

%extract data
[m,n] = size(sh);
ts = sh(2:m,1);
tsx = sh(1:m-1,:);
original = ts(length(sh)*0.7+1:end,:);

% Draw the original graphic of the stock price
figure;
plot(ts,'LineWidth',1);
title('Yahoo Stock Price(1996.4.12-2012.11.16) before mapping','FontSize',12);
grid on;

fprintf('Plot the stock price before mapping.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

%data preprocessing
ts = ts';
tsx = tsx';

% mapminmax is an mapping function in matlab
%Use mapminmax to do mapping
[TS,TSps] = mapminmax(ts);
% The scale of the data from 1 to 2
TSps.ymin = 1;
TSps.ymax = 2;
%normalization
[TS,TSps] = mapminmax(ts,TSps);

% plot the graphic of the stock price after mapping
figure;
plot(TS,'LineWidth',1);
title('Yahoo Stock price after mapping','FontSize',12);
grid on;

fprintf('\nPlot the stock price after mapping.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;


% Transpose the data in order to meet the requirement of libsvm
fprintf('\n Initializing.......\n');
TS = TS';

[TSX,TSXps] = mapminmax(tsx);
TSXps.ymin = 1;
TSXps.ymax = 2;
[TSX,TSXps] = mapminmax(tsx,TSXps);
TSX = TSX';

%split the data into training and testing
n1 = length(TS)*0.7;
train_label = TS(1:n1,:);
train_data = TSX(1:n1,:);
test_label = TS(n1+1:end,:);
test_data = TSX(n1+1:end,:);

fprintf('Begin the two round regressions to tune the parameter.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

% Find the optimize value of c,g paramter
% Approximately choose the parameters:
% the scale of c is 2^(-5),2^(-4),...,2^(10)
% the scale of g is 2^(-5),2^(-4),...,2^(5)
[bestmse,bestc,bestg] = svmregress(train_label,train_label,-5,10,-5,5,3,1,1,0.0005);

% Display the approximate result
disp('Display the approximate result');
str = sprintf( 'Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);
disp(str);

fprintf('\nFinish the first round tuning and begin the final round regression and print the final parameters.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

% Choose more precise parameter according to the graphic of previous step:
% the scale of c is 2^(0),2^(0.3),...,2^(10)
% the scale of g is 2^(-2),2^(-1.7),...,2^(3)
[bestmse,bestc,bestg] = svmregress(train_label,train_data,0,10,-2,3,3,0.3,0.3,0.0002);

disp('Display the final parameter result');
str = sprintf( 'Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);
disp(str);

fprintf('\nProgram paused. Press enter to continue.\n');
fprintf('Predict the stock price of the training data and compare to the grand truth.\n');
pause;

%Do training by using svmtrain of libsvm
cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg) , ' -s 3 -p 0.01'];
model = libsvmtrain(train_label,train_data,cmd);

%Do predicting by using svmpredict of libsvm
predict= libsvmpredict(test_label,test_data,model);
predict = mapminmax('reverse',predict,TSps);

% Display the result of SVM Regression
str = sprintf( 'MSE = %g R = %g%%',mse(2),mse(3)*100);
disp(str);

figure;
hold on;
plot(original,'LineWidth',1);
plot(predict,'r','LineWidth',1);
legend('Original Price','Predict Price','FontSize',10);
hold off;
grid on;
snapnow;