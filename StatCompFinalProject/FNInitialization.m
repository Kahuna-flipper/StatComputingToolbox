function [points,weights] = FNInitialization(input,num)
points = zeros(num,1);
[m,n] = size(input);
weights = zeros(num,n);
points(1) = randperm(m,1); % Initialization of first point 
for i=1:num-1
    distances=zeros(m,1);
    for j=1:m
        if(sum(ismember(points,j))~=0)
            continue
        end
        for l=1:i
            temp1= input(points(l),:);
            temp2= input(j,:);
            distances(j)=distances(j)+dtw(temp1,temp2);
        end
    end
    [~,index] = max(distances);
    points(i+1)=index;
end
for i=1:num
    weights(i,:)=input(points(i),:);
end
end


    


