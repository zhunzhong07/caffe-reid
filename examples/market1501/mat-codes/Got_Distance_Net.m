function dis = Got_Distance_Net(P, Q, label, class_num)
  caffe_dir = '../../../matlab';
  layer_name = 'score_dif';
  if exist([caffe_dir, '/+caffe'], 'dir')
	addpath(caffe_dir);
  else
	error('Can not find matlab caffe dir');
  end
  caffe.reset_all();
  caffe.set_mode_cpu();
  net_model = '../../../models/market1501/caffenet/deploy.proto';
  net_weights = '../../../models/market1501/caffenet/snapshot/caffenet.epoch.1_iter_8000.caffemodel';
  net = caffe.Net(net_model, net_weights, 'test');
  layer = net.layers(layer_name);
  weight = layer.params(1).get_data();
  assert (size(weight, 2) == 2);
  if (numel(layer.params) == 2), bias = layer.params(2).get_data();
  else, bias = zeros(2, 1); end
  [np, ~] = size(P); [nq, ~] = size(Q); assert(size(P,2) == size(Q,2));
  assert (numel(label) == nq);
  dis = zeros(np, class_num);
  fprintf('Got_Distance_Net Init Done : np : %d, nq : %d\n', np, nq);
  tic;
  for ii = 1:np
	cur_dis = repmat(P(ii,:), nq, 1) - Q;
    cur_dis = cur_dis .* cur_dis;
    cur_dis = cur_dis * weight + repmat(bias', nq, 1);
	cur_dis = norm_matrix(cur_dis, 'exp'); %% Softmax
	cur_sum = sum(cur_dis, 2);
	cur_dis = cur_dis ./ repmat(cur_sum, 1, 2);
	cur_dis = cur_dis(:, 2);
	for xlabel = 1:class_num
	  dis(ii, xlabel) = min(cur_dis(find(xlabel==label)));
	end
	if (rem(ii, 500) == 0 || ii == np), fprintf('%4d / %4d cost %.2f s\n', ii, np, toc); tic; end
  end

  caffe.reset_all();
  rmpath(caffe_dir);
end
