function [mse,bestc,bestg] = svmregress(train_label,train,cmin,cmax,gmin,gmax,v,cstep,gstep,msestep)
% SVMcgForClass
% Input:
% train_label:train_label, must meet the requirement of libsvm.
% train:train data, must meet the requirement of libsvm.
% cmin:the minimum value of c,c_min = 2^(cmin).default is -5
% cmax:the maximum value of c,c_max = 2^(cmax).default is 5
% gmin:the minimum value of g,g_min = 2^(gmin).default is -5
% gmax:the maximum value of g,g_min = 2^(gmax).default is 5
% v:the parameter of cross validation,means how many parts the data will be 
%   split intocross validation.default value is 3
% cstep:the step rate of c, default is 1
% gstep:the step rate of g, default is 1
% msestep:the step rate of the graphic od MSE.default is 20
% Ourput:
% bestacc:the best accuracy  during the process of Cross Validation
% bestc:the best value of c
% bestg:the best value of g

% about the parameters of SVMcgForRegress
if nargin < 10
    msestep = 0.1;
end
if nargin < 7
    msestep = 0.1;
    v = 3;
    cstep = 1;
    gstep = 1;
end
if nargin < 6
    msestep = 0.1;
    v = 3;
    cstep = 1;
    gstep = 1;
    gmax = 5;
end
if nargin < 5
    msestep = 0.1;
    v = 3;
    cstep = 1;
    gstep = 1;
    gmax = 5;
    gmin = -5;
end
if nargin < 4
    msestep = 0.1;
    v = 3;
    cstep = 1;
    gstep = 1;
    gmax = 5;
    gmin = -5;
    cmax = 5;
end
if nargin < 3
    msestep = 0.1;
    v = 3;
    cstep = 1;
    gstep = 1;
    gmax = 5;
    gmin = -5;
    cmax = 5;
    cmin = -5;
end
% X:c Y:g cg:mse
[X,Y] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);
[m,n] = size(X);
cg = zeros(m,n);
% record accuracy with different c & g,and find the best mse with the smallest c
bestc = 0;
bestg = 0;
mse = 10^10;
basenum = 2;
for i = 1:m
    for j = 1:n
        cmd = ['-v ',num2str(v),' -c ',num2str( basenum^X(i,j) ),' -g ',num2str( basenum^Y(i,j) ),' -s 3'];
        cg(i,j) = libsvmtrain(train_label, train, cmd);

        if cg(i,j) < mse
            mse = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end
        if ( cg(i,j) == mse && bestc > basenum^X(i,j) )
            mse = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end

    end
end
% draw the accuracy with different c & g
figure;
[C,h] = contour(X,Y,cg,0:msestep:1);
clabel(C,h,'FontSize',10,'Color','r');
xlabel('log2c','FontSize',10);
ylabel('log2g','FontSize',10);
grid on;
