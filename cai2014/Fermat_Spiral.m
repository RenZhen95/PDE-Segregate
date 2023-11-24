function [X,Y]= Fermat_Spiral
%Produce the data for the Fermat Sprial problem and plot the distribution
%[X,Y] = Fermat_Spiral;
%X is the sample matrix, in which each row refers to the sample and each
% column denotes the feature, Y is labelling vector of value -1, 1; 
%r = a theta^0.5; 
a = 4; b=4;sigma = 1.7; ns = 200
t1=linspace(0,2*pi,ns); r1=sqrt(t1); x1=-a*r1.* cos(t1); y1=-a*r1.*sin(t1);
x2=-b*r1.* cos(t1); y2=-b*r1.*sin(t1);
X = [x1' y1';
     -x2' -y2'] +sigma*randn(2*ns,2);

Y = [ones(ns,1);
      2*ones(ns,1)];
% NoisyFeature = randn(2*ns,nn);
% X = [X NoisyFeature];
if 1,
     plot(X(1:ns,1),X(1:ns,2),'bd',X(ns+1:end,1),X(ns+1:end,2),'r*');
    name =  'FermatSpiral.eps';
     print(gcf,'-depsc',name);
end

 
 
