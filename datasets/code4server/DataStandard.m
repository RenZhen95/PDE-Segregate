function A = DataStandard(A) 
A = double(A);
mu = mean(A,1);
mu = mu';
mode = ones(1,size(A,1));
mu2 = mu*mode;
stv = A-mu2';
s = std(A);
s = s';
mode = ones(1,size(A,1));
s2 = s*mode;
A = stv./s2';
end