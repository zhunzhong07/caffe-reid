function ANS = norm_matrix(matrix, str)
	[n, dim] = size(matrix);
	if (strcmp(str, 'exp'))
%	if (max(max(matrix)) > 10), matrix = matrix ./ max(max(matrix)); end
		temp = exp(matrix);
		dive = sum(temp, 2);
		ANS = temp ./ repmat(dive, 1, dim);
	elseif (strcmp(str, 'max'))
		ANS = matrix / max(max(matrix));
	else
		error(['Unknow type', str]);
	end

end
