function net = OneDTrain(input,weights,alpha,iterations,sigma,column)

[m,~] = size(input); % m is the number of inputs and n is the input dimension
[p,~] = size(weights); % p is the number of weights
distances = zeros(p,1);
for i=1:iterations
    for j=1:p
        temp1 = input(iterations,:);
        temp2 = weights(j,:);
        distances(p) = dtw(temp1,temp2);
    end
    [~,index] = min(distances); % stores best matching neuron
    weights(index,:) = weights(index,:) + alpha*(input(i,:)-weights(index,:));
    for r=1:sigma
        if(index-r>0)
            temp_index = index-r;
            weights(temp_index,:) = weights(temp_index,:) + alpha*exp(-(dtw(weights(temp_index,:),weights(index,:)))/(2*r^2))*(input(i,:)-weights(temp_index,:));
        end
        if(index+r<column)
            temp_index = index+r;
            weights(temp_index,:) = weights(temp_index,:) + alpha*exp(-(dtw(weights(temp_index,:),weights(index,:)))/(2*r^2))*(input(i,:)-weights(temp_index,:));
        end
    end
end
net = weights;    
end
