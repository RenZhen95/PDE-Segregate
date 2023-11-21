function [pred_label] = Difference_class_fast(testX, trainX, trainY, K, lambda)
%Difference form for alh
 
[N, D] = size(trainX);
Difference = (trainX - repmat(testX,N,1));
Dis = sum( ((Difference.^2))' );

    [Dis, ind] = sort(Dis);  
    ind_NN = ind(1:K);
    NN_label = trainY(ind_NN);
    if all(NN_label == NN_label(1))
        pred_label = NN_label(1); 
    else
        testX = testX';
        NN_data = trainX(ind_NN,:);
        classes_in_NN = unique(NN_label);
        nclass = length(classes_in_NN);
        Energy = zeros(1,nclass);
        Num_NN_All_Class= [];reg =[];
        for c = 1:nclass
            P = NN_data(NN_label == classes_in_NN(c),:);
            Num_NN_ClassC= size(P,1);
            if Num_NN_ClassC== 1 |norm(pdist(P))<1e-4
                Difference = (P - repmat(testX',size(P,1),1));
                Dis = sum( ((Difference.^2))');
                Energy(c) = sum(Dis(:)) + lambda; reg(c) = randn;
            else
        
             G_0 = (P - repmat(testX',Num_NN_ClassC,1));
             G =  G_0*G_0';
                if (lambda == 0), 
                       alpha_0 = pinv(G)*ones(Num_NN_ClassC,1);reg = [reg,alpha_0'*alpha_0;];
                       alpha = alpha_0/(alpha_0'*ones(Num_NN_ClassC,1));
                       
                else
                       alpha_0 = pinv(G+ lambda*trace(G)*eye(Num_NN_ClassC))*ones(Num_NN_ClassC,1);
                       reg = [reg,alpha_0'*alpha_0;];
                       alpha = alpha_0/(alpha_0'*ones(Num_NN_ClassC,1));
                end
             
                if lambda == 0, 
                        Energy(c) =  alpha'*G*alpha;
                   else Energy(c) = alpha'*G*alpha + lambda*(alpha'*alpha);
                end
            end
            Num_NN_All_Class= [Num_NN_All_Class Num_NN_ClassC]; 
        end
        ind_bestclass = find(Energy == min(Energy));
        if length(ind_bestclass) > 1
            ind_bestclass = ind_bestclass( Num_NN_All_Class(ind_bestclass) == max(Num_NN_All_Class(ind_bestclass))  );
            length_ind = length(ind_bestclass);
            if length_ind > 1
                [tp,ind_bestclass] = min(reg); % Take the sample of smaller reguliazation
            end
        end
        pred_label = classes_in_NN(ind_bestclass); 
    end
%save Inversion Inversion


