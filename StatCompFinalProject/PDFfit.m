function [pdfs,ll] = PDFfit(data)

%% For the first response :
    [m,~]= size(data);


    x=linspace(min(data),max(data),m);


    pd_norm=fitdist(data,'Normal'); % MLE for normal disribution

    %% Extracting normal dist parameters mu and sigma :
    normal_mu = pd_norm.mu;
    normal_sigma = pd_norm.sigma;

    pd_gamma = fitdist(data,'Gamma'); % MLE for gamma distribution
    %% Extracting gamma dist parameters :
    a = pd_gamma.a;
    b = pd_gamma.b;

    pd_wei = fitdist(data,'Weibull'); % MLE for weibull distribution
    %% Extracting Weibull dist parameters :
    A = pd_wei.A;
    B = pd_wei.B;
    % 
    %% Log likelihood calculation
    logl_norm=sum(log(pdf(pd_norm,data)));
    logl_gamma=sum(log(pdf(pd_gamma,data)));
    logl_wei=sum(log(pdf(pd_wei,data)));

    %% AIC calculation

    AIC_norm = -2*logl_norm + 2*2 + (2*2*(2+1))/(m-2-1);
    AIC_gamma = -2*logl_gamma + 2*2 + (2*2*(2+1))/(m-2-1);
    AIC_wei = -2*logl_wei + 2*2 + (2*2*(2+1))/(m-2-1);

    %% Del AIC calculastion
    AIC = [AIC_norm;AIC_gamma;AIC_wei];
    del_AIC_norm = AIC_norm-min(AIC);
    del_AIC_gamma = AIC_gamma-min(AIC);
    del_AIC_wei = AIC_wei-min(AIC);


    %% Plotting data and fitted distributions
    figure();
    subplot(3,1,1)
    plot(x,pdf(pd_norm,x),'color','r','LineWidth',2)
    hold on
    plot(x,pdf(pd_gamma,x),'color','g','LineWidth',2)
    hold on
    plot(x,pdf(pd_wei,x),'color','b','LineWidth',2)
    hold on
    histogram(data,'Normalization','pdf','NumBins',10,'FaceColor','b','FaceAlpha',0.3)
    legend(strcat('Normal, LL = ', num2str(logl_norm)),strcat('Gamma, LL = ', num2str(logl_gamma)),strcat('Wei, LL= ', num2str(logl_wei)))
    title('Plotting fits for data');
    subplot(3,1,2)
    histogram(data,'Normalization','pdf','NumBins',10,'FaceColor','b','FaceAlpha',0.3)
    title('Centroid');
    subplot(3,1,3)
    plot(data,'b','Linewidth',2)
    title('Signalling profile of centroid');
    %% PP plot for CDF 

    [N,~] = histcounts(data);
    [~,k] = size(N);
    emp_cdf = zeros(k,1);
    curr_sum=0;

    for i=1:k
        for j=1:i
            curr_sum = curr_sum + N(j);
        end
        emp_cdf(i) = curr_sum/sum(N);
        curr_sum=0;
    end

    y=linspace(min(data),max(data),k);
    y1 = linspace(0,1,k);
    norm_cdf = cdf(pd_norm,y);
    gamma_cdf = cdf(pd_gamma,y);
    wei_cdf = cdf(pd_wei,y);

    %% Calcularing rmsd for the three distributions
    rm_norm=0;
    rm_gamma=0;
    rm_wei=0;
    for i=1:k
        rm_norm = rm_norm + (emp_cdf(i)-norm_cdf(i))^2/k;
        rm_gamma = rm_gamma + (emp_cdf(i)-gamma_cdf(i))^2/k;
        rm_wei = rm_wei + (emp_cdf(i)-wei_cdf(i))^2/k;
    end

    %% Plotting PP Plot and rmsd
%     figure(2);
%     plot(emp_cdf,norm_cdf,'ro');
%     hold on
%     plot(emp_cdf,gamma_cdf,'bx');
%     hold on
%     plot(emp_cdf,wei_cdf,'g+');
%     hold on
%     plot(y1,y1,'-','Linewidth',3);
%     legend(strcat('Normal cdf, rmsd = ', num2str(rm_norm)),strcat('Gamma cdf, rmsd = ', num2str(rm_gamma)),strcat('Wei cdf, rmsd = ', num2str(rm_wei)))
%     title('PP plot for CDF');

    %% Plotting theorectical and empirical cdf
%     figure(3)
%     plot(emp_cdf,'b','Linewidth',3);
%     hold on
%     plot(norm_cdf,'r','Linewidth',3);
%     hold on
%     plot(gamma_cdf,'g','Linewidth',3);
%     hold on
%     plot(wei_cdf,'m','Linewidth',3);
%     legend('Empirical cdf','Normal cdf','Gamma cdf','Weibull cdf');
%     title('Empirical and Theoretical CDF');

    %% Calculating KL divergence
    emp_pdf = zeros(k,1);
    for i=1:k
        emp_pdf(i)=N(i)/sum(N);
    end


    y=linspace(min(data),max(data),k);
    norm_pdf = pdf(pd_norm,y);
    gamma_pdf = pdf(pd_gamma,y);
    wei_pdf = pdf(pd_wei,y);

    kl_norm=0;
    kl_gamma=0;
    kl_wei=0;

    for i=1:k
        if(emp_pdf(i)~=0)
            kl_norm = kl_norm + norm_pdf(i)*log(norm_pdf(i)/emp_pdf(i));
            kl_gamma = kl_gamma + gamma_pdf(i)*log(gamma_pdf(i)/emp_pdf(i));
            kl_norm = kl_norm + wei_pdf(i)*log(wei_pdf(i)/emp_pdf(i));
        end
    end
    
    pdfs = [pd_norm;pd_gamma;pd_wei];
    ll = [logl_norm,logl_gamma,logl_wei];
end