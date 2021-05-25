function dEuc = EuclideanDistance(sample1, sample2)


for i=1:size(sample1, 1)

    dist = sqrt(sum((sample1(i,:) - sample2(i,:)) .^ 2));

end
dEuc = dist;

end

