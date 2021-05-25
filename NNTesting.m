function prediction = NNTesting(testImage, modelNN)

    noOfTrainingImages = size(modelNN.neighbours, 1);

    distances = zeros(noOfTrainingImages, 1);

    for i = 1:noOfTrainingImages

       euc = EuclideanDistance(testImage(1,:), modelNN.neighbours(i,:));

       distances(i,1) = euc;

    end

    [M,I] = min(distances);



    prediction = modelNN.labels(I,1);
end

