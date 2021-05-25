function prediction = KNNTesting(testImage, modelNN, K)
   
    noOfTrainingImages = size(modelNN.neighbours, 1);

    distances = zeros(noOfTrainingImages, 1);

    for i = 1:noOfTrainingImages

       euc = EuclideanDistance(testImage(1,:), modelNN.neighbours(i,:));

       distances(i,1) = euc;

    end
    
    [B, I] = sort(distances);
    labels = zeros(K, 1);
    
    for j =1:K
        labels(j,1) = modelNN.labels(I(j))
    end
    M= mode(labels)
    prediction = M
    
    
end

