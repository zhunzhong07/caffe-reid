function ANS = cal_score_from_dis(distance, label, class_num)
  [n, m] = size(distance);
  assert (m == numel(label));
  %MX = max(distance, [], 2);
  ANS = zeros(n, class_num);
  for idx = 1:class_num
	index = find(label == idx);
	ANS(:, idx) = min(distance(:, index), [], 2);
	%ANS(:, idx) = mean(distance(:, index), 2);
  end
%  ANS = ANS ./ repmat(max(ANS, [], 2), 1, class_num);
end
