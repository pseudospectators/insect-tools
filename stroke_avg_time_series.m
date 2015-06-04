function [tavg,favg]= stroke_avg_time_series( t, f, tstroke )
%% stroke_avg_time_series( t, f, tstroke )
% take the signal f(t) and return the stroke-averages for each stroke 
% the signal f(t) does not need to be equidistant
% tstroke is optional (default=1)

if (nargin==2)
    tstroke=1;
end

tmax = t(end);
n = floor(tmax/tstroke);

% fprintf('we have %i complete strokes\n', n)

tavg=zeros(n,1);
favg=zeros(n,1);

for i = 1:n
   t1 = (i-1)*tstroke;
   t2 = (i)*tstroke;
   
   tavg(i) = 0.5*(t1+t2);
   favg(i) = stroke_avg(t,f,t1,t2,'linear');
end
