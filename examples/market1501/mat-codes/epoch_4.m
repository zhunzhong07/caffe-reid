clc;clear all;close all;
[Epoch_1,  label_1]  = textread('../lists/train.epoch.1', '%s %d');
[Epoch_1L, label_1L] = textread('../lists/train.epoch.1L', '%s %d');
label_1L = label_1L + 1;
label_1  = label_1 + 1;
%score_1L = importdata('../datamat/train.1L.score.mat');
%score_1  = importdata('../datamat/train.1.score.mat');
%fc7_1L = importdata('../datamat/train.1L.fc7.mat');
%fc7_1  = importdata('../datamat/train.1.fc7.mat');
score_1L = importdata('../datamat/train.1L.epoch4.score.mat');
score_1  = importdata('../datamat/train.1.epoch4.score.mat');
fc7_1L = importdata('../datamat/train.1L.epoch4.fc7.mat');
fc7_1  = importdata('../datamat/train.1.epoch4.fc7.mat');

assert (size(fc7_1, 2) == 4096);
unique_label = unique(label_1);
count_per_class = zeros(numel(unique_label), 1);
class_num = numel(unique_label);
ratio = 0.8;
for label = 1:numel(unique_label)
    count_per_class(label) = numel(find(label_1==label)) + numel(find(label_1L==label));
    count_per_class(label) = ceil(count_per_class(label) * ratio) - numel(find(label_1==label));
end


distance = cal_dis(fc7_1L', fc7_1', 'Euc');
distance = cal_score_from_dis(distance, label_1, class_num);
distance = norm_matrix(distance, 'exp');
%distance = Got_Distance_Net(fc7_1L, fc7_1, label_1, class_num);

[Dmx_score, Dmx_label] = max(distance, [], 2);
[Dmx_class, Dmx_cls_i] = max(distance, [], 1);

[Smx_score, Smx_label] = max(score_1L, [], 2);
[Smx_class, Smx_cls_i] = max(score_1L, [], 1);

weight = 0;
[Mmx_score, Mmx_label] = max(score_1L + weight * distance, [], 2);
[Mmx_class, Mmx_cls_i] = max(score_1L + weight * distance, [], 1);

select_per_class = cell(numel(unique_label), 1);
label_per_class = cell(numel(unique_label), 1);
acc_zero = [];
for label = 1:numel(unique_label)
	idx = find(Mmx_label == label);
	score = Mmx_score(idx);
	[~, args] = sort(score, 'descend');
	select_M = min(count_per_class(label), numel(args));
	select_M = idx( args(1:select_M) );

	idx = find(Smx_label == label);
	score = Smx_score(idx);
	[~, args] = sort(score, 'descend');
	select_S = min(count_per_class(label), numel(args));
	select_S = idx( args(1:select_S) );

	select_per_class{label} = intersect(select_M, select_S);
	%select_per_class{label} = union(select_M, select_S);

	if (numel(select_per_class{label}) == 0)
		Stop = Smx_cls_i(label);
		Dtop = Dmx_cls_i(label);
		if (Smx_label(Stop) == label)
			select_per_class{label} = Stop;
		elseif (Dmx_label(Dtop) == label)
			select_per_class{label} = Dtop;
		end
	end

	if (numel(select_per_class{label}) > 0)
		temp = select_per_class{label};
		temp = temp(find(Mmx_score(temp)>0.4));
	 	select_per_class{label} = temp;
	end
	if (numel(select_per_class{label}) == 0)
		select_per_class{label} = select_S;
	end
	%if (numel(select_per_class{label}) == 0)
	%select_per_class{label} = Smx_cls_i(label);
	%end

	true_label = label_1L(select_per_class{label});
	label_per_class{label} = zeros(numel(true_label), 1) + label;

	fprintf('Label %3d, generate %2d / %2d pesudo label, ok : %2d, accuracy : %.3f, score : %.3f\n', label, numel(select_per_class{label}), numel(find(label_1L==label)), numel(find(true_label==label)), mean(true_label==label), min(Smx_score(select_per_class{label})));

	if (numel(find((true_label==label))) == 0) 
		acc_zero(end+1) = label;
	end
end
select_all = cat(1, select_per_class{:});
label_all = cat(1, label_per_class{:});
fprintf('Total generate %3d / %3d pesudo label, accuracy: %.5f  || zero_acc : %d\n', numel(select_all), numel(label_1L), mean(label_1L(select_all) == label_all), numel(acc_zero));

%% Got_Distance_Net Min : accuracy: 0.89017  ||  un_select : 5 || zero_acc : 56
%% Normalize            : accuracy: 0.91474  || zero_acc : 38

list_file = fopen('../lists/train.epoch.5', 'w');

select_all = [Epoch_1L(select_all); Epoch_1];
label_all  = [label_all ; label_1];
for i = 1:numel(label_all)
	fprintf(list_file, '%s %d\n', select_all{i}, label_all(i)-1);
end
fclose(list_file);
