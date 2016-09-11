function sift()

%target folder
conf.calDir = 'C:\Users\User\Desktop\FYP\images' ;
conf.dataDir = 'C:\Users\User\Desktop\FYP\data' ;
%train and test number of images
conf.numTrain = 250 ;
conf.numTest = 250 ;
conf.numClasses = 2 ;
%gaussian mode
conf.numWords = 256 ;
%conf.numSpatialX = [2 4] ;
%conf.numSpatialY = [2 4] ;
conf.quantizer = 'fv' ;
conf.svm.C = 10 ;

%conf.svm.solver = 'sdca' ;
conf.svm.solver = 'sgd' ;
%conf.svm.solver = 'liblinear' ;

conf.svm.biasMultiplier = 1 ;
%conf.phowOpts = {'Sizes', 4, 'Step', 2} ;
conf.clobber = false ;
conf.randSeed = 1 ;

conf.vocabPath = fullfile(conf.dataDir, ['vocab.mat']) ;
conf.histPath = fullfile(conf.dataDir, ['hists.mat']) ;
conf.modelPath = fullfile(conf.dataDir, ['model.mat']) ;
conf.resultPath = fullfile(conf.dataDir, ['result']) ;

randn('state',conf.randSeed) ;
rand('state',conf.randSeed) ;
vl_twister('state',conf.randSeed) ;

% --------------------------------------------------------------------
%                                                           Setup data
% --------------------------------------------------------------------
classes = dir(conf.calDir) ;
classes = classes([classes.isdir]) ;
classes = {classes(3:conf.numClasses+2).name} ;

images = {} ;
imageClass = {} ;

for ci = 1:length(classes)
  ims = dir(fullfile(conf.calDir, classes{ci}, '*.jpg'))' ;
  ims = vl_colsubset(ims, conf.numTrain + conf.numTest) ;
  ims = cellfun(@(x)fullfile(classes{ci},x),{ims.name},'UniformOutput',false) ;
  images = {images{:}, ims{:}} ;
  imageClass{end+1} = ci * ones(1,length(ims)) ;
end

selTest1 = (1:250);
selTest2 = (501:750);
selTest = horzcat(selTest1, selTest2);
%selTrain = find(mod(0:length(images)-1, conf.numTrain+conf.numTest) < conf.numTrain) ;
%selTest = setdiff(1:length(images), selTrain) ;
selTrain = setdiff(1:length(images), selTest) ;
imageClass = cat(2, imageClass{:}) ;

model.classes = classes ;
%model.phowOpts = conf.phowOpts ;
%model.numSpatialX = conf.numSpatialX ;
%model.numSpatialY = conf.numSpatialY ;
model.quantizer = conf.quantizer ;
% model.vocab = [] ;
model.means = [];
model.covariances = [];
model.priors = [];
model.w = [] ;
model.b = [] ;
model.classify = @classify ;

% --------------------------------------------------------------------
%                                                     Train vocabulary
% --------------------------------------------------------------------

if ~exist(conf.vocabPath) || conf.clobber

  % Get some SIFT descriptors to train the dictionary
  selTrainFeats = vl_colsubset(selTrain, 500) ;
  d = {} ;
  
  parfor ii = 1:length(selTrainFeats)
      im = imread(fullfile(conf.calDir, images{selTrainFeats(ii)})) ;
      im = single(rgb2gray(im)) ;
      %colormap(gray(256))
      %im = standarizeImage(im) ;
    [drop, d{ii}] = vl_sift(im); 
  end
  
   d = vl_colsubset(cat(2, d{:}), 10e4) ;
   d = single(d) ;
   
  % Quantize the descriptors to get the visual words
  % vocab = vl_kmeans(descrs, conf.numWords, 'verbose', 'algorithm', 'elkan', 'MaxNumIterations', 50) ;
  % save(conf.vocabPath, 'vocab') ;
  
    [means,covariances,priors] = ...
    vl_gmm(d, conf.numWords, 'verbose','NumRepetitions', 1) ; 
    save(conf.vocabPath,'priors', 'means','covariances') ;
else
  load(conf.vocabPath) ;
end

% model.vocab = vocab ;
model.means = single(means);
model.covariances = single(covariances);
model.priors = single(priors);
% if strcmp(model.quantizer, 'kdtree') //ctrl r
%   model.kdtree = vl_kdtreebuild(vocab) ;
% end

% --------------------------------------------------------------------
%                                           Compute spatial histograms
% --------------------------------------------------------------------

if ~exist(conf.histPath) || conf.clobber
  fv = {} ;
  parfor ii = 1:length(images)
    fprintf('Processing %s (%.2f %%)\n', images{ii}, 100 * ii / length(images)) ;
    im = imread(fullfile(conf.calDir, images{ii})) ;
    fv{ii} = getImageDescriptor(model, im);
  end

  fv = cat(2, fv{:}) ;
  save(conf.histPath, 'fv') ;
else
  load(conf.histPath) ;
end

% --------------------------------------------------------------------
%                                                  Compute feature map
% --------------------------------------------------------------------

psix = fv ;

% --------------------------------------------------------------------
%                                                            Train SVM
% --------------------------------------------------------------------
if ~exist(conf.modelPath) || conf.clobber
  switch conf.svm.solver
    case {'sgd', 'sdca'}
      lambda = 1 / (conf.svm.C *  length(selTrain)) ;
      w = [] ;
      parfor ci = 1:length(classes)
        perm = randperm(length(selTrain)) ;
        fprintf('Training model for class %s\n', classes{ci}) ;
        y = 2 * (imageClass(selTrain) == ci) - 1 ;
        [w(:,ci) b(ci) info] = vl_svmtrain(psix(:, selTrain(perm)), y(perm), lambda, ...
          'Solver', conf.svm.solver, ...
          'MaxNumIterations', 50/lambda, ...
          'BiasMultiplier', conf.svm.biasMultiplier, ...
          'Epsilon', 1e-3);
      end

    case 'liblinear'
      svm = train(imageClass(selTrain)', ...
                  sparse(double(psix(:,selTrain))),  ...
                  sprintf(' -s 3 -B %f -c %f', ...
                          conf.svm.biasMultiplier, conf.svm.C), ...
                  'col') ;
      w = svm.w(:,1:end-1)' ;
      b =  svm.w(:,end)' ;
  end

  model.b = conf.svm.biasMultiplier * b ;
  model.w = w ;

  save(conf.modelPath, 'model') ;
else
  load(conf.modelPath) ;
end

% --------------------------------------------------------------------
%                                                Test SVM and evaluate
% --------------------------------------------------------------------

% Estimate the class of the test images
scores = model.w' * psix + model.b' * ones(1,size(psix,2)) ;
[drop, imageEstClass] = max(scores, [], 1) ;

% Compute the confusion matrix
idx = sub2ind([length(classes), length(classes)], ...
              imageClass(selTest), imageEstClass(selTest)) ;
confus = zeros(length(classes)) ;
confus = vl_binsum(confus, ones(size(idx)), idx) ;

% Plots
figure(1) ; clf;
subplot(1,2,1) ;
imagesc(scores(:,[selTrain selTest])) ; title('Scores') ;
set(gca, 'ytick', 1:length(classes), 'yticklabel', classes) ;
subplot(1,2,2) ;
imagesc(confus) ;
title(sprintf('Confusion matrix (%.2f %% accuracy)', ...
              100 * mean(diag(confus)/conf.numTest) )) ;
print('-depsc2', [conf.resultPath '.ps']) ;
save([conf.resultPath '.mat'], 'confus', 'conf') ;

% -------------------------------------------------------------------------
function im = standarizeImage(im)
% -------------------------------------------------------------------------

im = im2single(im) ;
if size(im,1) > 480, im = imresize(im, [480 NaN]) ; end

% -------------------------------------------------------------------------
function fv = getImageDescriptor(model, im)
% -------------------------------------------------------------------------
im = standarizeImage(im) ;
%width = size(im,2) ;
%height = size(im,1) ;
%numWords = size(model.means, 2) ;

% get SIFT features
im = single(rgb2gray(im)) ;
%colormap(gray(256))
[f, d] = vl_sift(im); 

% quantize local descriptors into visual words
switch model.quantizer
  case 'vq'
    [drop, binsa] = min(vl_alldist(model.vocab, single(descrs)), [], 1) ;
  case 'kdtree'
    binsa = double(vl_kdtreequery(model.kdtree, model.vocab, ...
                                  single(descrs), ...
                                  'MaxComparisons', 50)) ;
  case 'fv'
    fv = vl_fisher(single(d),...
                      model.means,...
                      model.covariances,...
                      model.priors,...
                      'Improved');     
end

%{
for i = 1:length(model.numSpatialX)
  binsx = vl_binsearch(linspace(1,width,model.numSpatialX(i)+1), frames(1,:)) ;
  binsy = vl_binsearch(linspace(1,height,model.numSpatialY(i)+1), frames(2,:)) ;

  % combined quantization
  bins = sub2ind([model.numSpatialY(i), model.numSpatialX(i), numWords], ...
                 binsy,binsx,binsa) ;
  hist = zeros(model.numSpatialY(i) * model.numSpatialX(i) * numWords, 1) ;
  hist = vl_binsum(hist, ones(size(bins)), bins) ;
  hists{i} = single(hist / sum(hist)) ;
end
hist = cat(1,hists{:}) ;
hist = hist / sum(hist) ;
%}

% -------------------------------------------------------------------------
function [className, score, scores] = classify(model, im)
% -------------------------------------------------------------------------

%hist = getImageDescriptor(model, im) ;
fv = getImageDescriptor(model, im) ;
psix = fv;
scores = model.w' * psix + model.b' ;
[score, best] = max(scores) ;
className = model.classes{best} ;
