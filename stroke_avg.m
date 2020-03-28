function avg = stroke_avg(t,f,t1,t2,method)
%% avg = stroke_avg(t,f,t1,t2)
% averages the signal f gives at times t between t1 and t2
% values are extrapolated with zeros.
% The signal f is interpolated to an equidistant vector; input field
% can thus be arbitrarily spaced

if (nargin==4)
    method='spline';
end

dt = (t(end)-t(1)) / length(t);

nt = ceil ( (t2-t1) /dt ); 

T_interval = t1 + (t2-t1) * (0:nt-1)/(nt-1);

avg = mean( interp1(t,f,T_interval,method,0) );