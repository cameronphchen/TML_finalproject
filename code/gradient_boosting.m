% standard gradient boosting
% by Cameron P.H. Chen @Princeton 
% contact: pohsuan [at] princeton [dot] edu

% loss function: l2 loss function
% hypothesis space : step functions 0->1, 0->-1, -1->0, 1->0;

function [alpha_result] = gradient_boosting(data,num_iter)





d_length = size(data,1);
d_min  = min(data(:,1));
d_max  = max(data(:,1));
d_range = d_max -d_min +1;
% alpha (location of activation, type of activation, iterations)
% type of activation 
% 1:0->1, 2:0->-1, 3:1->0, 4:-1->0;
alpha = zeros(d_length,4,num_iter+1);

r = zeros(d_range,num_iter+1);

for i=2:num_iter+1
  %fprintf('%d th iteration\n',i-1)
  %alpha
  %h
  %step_function(data(:,1),h(:,i-1),d_range)

  F = zeros(d_range,1);
  for k = 1:d_length
    for s = 1:4
      if alpha(k,s,i-1) ~=0
        step_func = step_function(data(k,1),s,d_range);
        F = F + step_func/norm(step_func(data(:,1)))*alpha(k,s,i-1);
      end
    end
  end

  for k=1:d_length
    r(data(k,1),i) = data(k,2) - F(data(k,1));
  end

  if norm(r(:,i))<0.00001
    alpha(:,:,num_iter+1)=alpha(:,:,i-1);
    fprintf('residual = 0, break early at %d-th iteration\n',i)
    break  
  end

  %argmax h
  %step function is defined on the location of data
  %t_max is the location where step happens, also indicate which the step 
  %function it is

  t_max=0; 
  t_max_type =0;
  tmp_max =-inf;
  for t=1:d_length
    for s=1:4
      %fprintf('t:%d,s:%d',t,s)
      step_func = step_function(data(t,1),s,d_range);
      norm_step = step_func/norm(step_func(data(:,1)));
      tmp = r(data(:,1),i)'*norm_step(data(:,1));
      if (tmp > tmp_max) 
        tmp_max = tmp;
        t_max = t;
        t_max_type = s;
      end
    end
  end


  %epsilon
  step_func = step_function(data(t_max,1),t_max_type,d_range);
  norm_step = step_func/norm(step_func(data(:,1))); 
  epsilon = r(data(:,1),i)'*norm_step(data(:,1));

  %update alpha
  alpha(:,:,i) = alpha(:,:,i-1);
  alpha(t_max,t_max_type,i) = alpha(t_max,t_max_type,i) + epsilon;
end


F_end = 0;
for k = 1:d_length
  for s=1:4
    if alpha(k,s,end) ~= 0
      step_func = step_function(data(k,1),s,d_range);
      F_end = F_end + step_func/norm(step_func(data(:,1)))*alpha(k,s,end);
    end
  end
end

F_end;
r_end = zeros(d_range,1);
for k=1:d_length
  r_end(data(k,1)) = data(k,2) - F_end(data(k,1));
end

alpha_result = alpha(:,:,end);
fprintf('result')
%r
%alpha


figure
hold on
grid on
scatter(data(:,1),data(:,2),1000*ones(d_length,1),'.');
plot(d_min:d_max, F_end, 'r', 'Linewidth',2);

