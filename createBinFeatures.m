function BinX = createBinFeatures(x)

%%% is a col vector of features with int values 
%%% Return a matrix of features with bin values
%%%Add 1 bc min val of features is 0 and indexing on octave starts from 1

m = size(x);

BinX = zeros(m, (max(x)+1));


for i=1:m
    BinX(i, x(i)+1) = 1;
end
