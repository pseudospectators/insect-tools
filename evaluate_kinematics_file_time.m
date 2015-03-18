function [phi,alpha,theta] = evaluate_kinematics_file_time(time,kine)
%% [phi,alpha,theta] = evaluate_kinematics_file_time(time,kine)
% evaluate the kinematics "kine" (read_kinematics_file)
% on the time vector t

phi   = zeros(size(time));
alpha = zeros(size(time));
theta = zeros(size(time));

for it = 1:length(time)
   phi(it)   = Fseriesval( [kine.a0_phi; kine.ai_phi], kine.bi_phi, time(it) );
   alpha(it) = Fseriesval( [kine.a0_alpha; kine.ai_alpha], kine.bi_alpha, time(it) );
   theta(it) = Fseriesval( [kine.a0_theta; kine.ai_theta], kine.bi_theta, time(it) );
end
