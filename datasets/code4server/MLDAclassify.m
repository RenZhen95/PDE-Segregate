function macc =MLDAclassify(X,Y,CVO,method)
switch method
    case 'cv'
        for fold_idx = 1:CVO.NumTestSets
            train_id = find(CVO.training(fold_idx));
            test_id = find(CVO.test(fold_idx));
            trainSamples = X(train_id,:);
            trainClasses = Y(train_id);
            testClasses = Y(test_id);
            testSamples = X(test_id,:);
            %************************* MultiClass LDA ***************************************
            mLDA = LDA(trainSamples, trainClasses);
            mLDA.Compute();
            %dimension of a samples is < (mLDA.NumberOfClasses-1) so following line cannot be executed:
            %transformedSamples = mLDA.Transform(meas, mLDA.NumberOfClasses - 1);
            transformedTrainSamples = mLDA.Transform(trainSamples, 1);
            transformedTestSamples = mLDA.Transform(testSamples, 1);
            
            %************************* MultiClass LDA ***************************************
            
            calculatedClases = knnclassify(transformedTestSamples, transformedTrainSamples, trainClasses);
            
            similarity = [];
            for i = 1 : length(testClasses)
                similarity(i) = ( testClasses(i) == calculatedClases(i) );
            end
            
            accuracy(fold_idx) = sum(similarity) / length(testClasses);
        end
        macc = mean(accuracy);
        
    case 'loo'
        [ns,nf]  = size(X);
        ind = 1:ns;
        for fi = 1:ns
            train_id = setdiff(ind,fi);;
            test_id = fi;
            trainSamples = X(train_id,:);
            trainClasses = Y(train_id);
            testClasses = Y(test_id);
            testSamples = X(test_id,:);
            %************************* MultiClass LDA ***************************************
            mLDA = LDA(trainSamples, trainClasses);
            mLDA.Compute();
            %dimension of a samples is < (mLDA.NumberOfClasses-1) so following line cannot be executed:
            %transformedSamples = mLDA.Transform(meas, mLDA.NumberOfClasses - 1);
            transformedTrainSamples = mLDA.Transform(trainSamples, 1);
            transformedTestSamples = mLDA.Transform(testSamples, 1);
            
            %************************* MultiClass LDA ***************************************
            calculatedClases = knnclassify(transformedTestSamples, transformedTrainSamples, trainClasses);
            tp(fi)  = calculatedClases==testClasses;
        end
        macc   = sum(tp)/ns;
end