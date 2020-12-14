function [labels,final_weights,TRAIN] = Neurons(num_centroids)

   %% Load train.tsv file from directory

    temp = xlsread('Data1.xlsx',2);
    temp = temp(1:100,:);
    TRAIN = temp;
    [m,n] = size(TRAIN);
    %shuffle = randperm(m,m);
%     w = gausswin(1);
%     w = w/sum(w);
    for i=1:n
        %TRAIN(:,n) = smoothdata(TRAIN(:,n));
        TRAIN(:,n) = movmean(TRAIN(:,n),1);
        %TRAIN(:,n) = wdenoise(TRAIN(:,n));
        %TRAIN(:,i) = detrend(TRAIN(:,i),3);
        %TRAIN(:,i) = filter(w,1,TRAIN(:,i));
    end

    tic
%     for i=1:m
%         TRAIN(i,:)=TRAIN(shuffle(i),:);
%     end
   

    %% Cutting size

    norm_train = TRAIN(:,1:n);

    norm_train = norm_train';
    [m,n] = size(norm_train);

    %% Z Normalization
%     for i=1:m
%         norm_train(i,:) = (norm_train(i,:)-mean(norm_train(i,:)))/std(norm_train(i,:));
%     end

    labels = zeros(m,1);
   


    %% Calculating results with minimum alpha 
    alpha = 0.8;
    [~,final_weights] = FNInitialization(norm_train,num_centroids);
        for epochs= 1:10
            final_net = OneDTrain(norm_train,final_weights,alpha,m,0,num_centroids);
            final_weights = final_net;
            alpha = alpha/epochs;
        end

        for i= 1:m
            dist = zeros(num_centroids,1);
            for j= 1:num_centroids
                dist(j) = dtw(norm_train(i,:),final_net(j,:));
            end
            [~,labels(i)] = min(dist);
        end
end