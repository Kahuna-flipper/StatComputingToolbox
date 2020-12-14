%% Fitting probability distributions

[labels,final_weights,TRAIN] = Neurons(3);


%% Calculating pdfs
[pdfs1,ll1] = PDFfit(final_weights(1,:)');
[pdfs2,ll2]= PDFfit(final_weights(2,:)');
[pdfs3,ll3] = PDFfit(final_weights(3,:)');







