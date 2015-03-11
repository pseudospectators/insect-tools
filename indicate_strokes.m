function indicate_strokes(varargin)
%% function indicate_strokes(varargin)
% add shaded areas in the background of existing plots, usually to 
% highlight up/downstrokes. 
% -------------------------------------------------------------------------
% Inputs:
%   none
%   Optional:
%     'figure' followed by an integer: number of figure to use. default: gcf
%     'color' followed by vector: color of shaed areas
%     'frame' yes/no: put frames around shaded areas
%     't_begin' array of times beginning of downstroke
%     't_reversal' array of times of stroke reversals
% If t_begin and t_reversal are missing, unit period time and symmetric up/
% downstroke are assumed
% -------------------------------------------------------------------------
% Output:
%   none
% -------------------------------------------------------------------------


%% define default values here:
fig = gcf;
color = [0.85 0.85 0.85];
frame = 'yes';
t_begin = 0;
t_reversal = 0;

%% read in optional arguments
i = 1;
while i <= length(varargin)
   switch varargin{i}
       case 'figure'
           % check if we're not at the end of the parameter list
           assert(i+1<=numel(varargin),'missing an argument!');
           i = i+1;
           % check if next argument is numeric
           if (~isnumeric(varargin{i}))
               error(['Parameter value for ' varargin{i-1} ' must be numeric!']);
           end
           % take that value
           fig = varargin{i};
       case 'color'
           % check if we're not at the end of the parameter list
           assert(i+1<=numel(varargin),'missing an argument!');
           i = i+1;
           % check if next argument is numeric
           if (~isnumeric(varargin{i}))
               error(['Parameter value for ' varargin{i-1} ' must be numeric!']);
           end
           color = varargin{i};
       case 'frame'
           assert(i+1<=numel(varargin),'missing an argument!');
           i = i+1;
           if (~ischar(varargin{i}))
               error(['Parameter value for ' varargin{i-1} ' must be string!']);
           end
           frame = varargin{i};
       case 't_begin'
           assert(i+1<=numel(varargin),'missing an argument!');
           i = i+1;
           if (~isnumeric(varargin{i}))
               error(['Parameter value for ' varargin{i-1} ' must be numeric!']);
           end
           t_begin = varargin{i};  
       case 't_reversal'
           assert(i+1<=numel(varargin),'missing an argument!');
           i = i+1;
           if (~isnumeric(varargin{i}))
               error(['Parameter value for ' varargin{i-1} ' must be numeric!']);
           end
           t_reversal = varargin{i};
       otherwise
           error(['unkown parameter:' varargin{i}])
   end
   i=i+1;
end

% determine what the current axis are
figure(fig)
a = axis;
tmin = a(1);
tmax = a(2);
fmin = a(3);
fmax = a(4);

% determine beginnings of shaded areas (beginning of downstroke)
if (length(t_begin)>1)
    tdown = t_begin;
else
    tdown = 0:tmax-1;
end

% determine end of shaded areas (reversal timing)
if (length(t_reversal)>1)
    treversal = t_reversal;
else
    treversal = tdown + 0.5;
end

for i = 1:length(tdown)
    if (strcmp(frame,'yes'))
        patch( [tdown(i) treversal(i) treversal(i) tdown(i)], [fmin fmin fmax fmax],...
             [-1 -1 -1 -1], color);
    else
        patch( [tdown(i) treversal(i) treversal(i) tdown(i)], [fmin fmin fmax fmax],...
             [-1 -1 -1 -1], color,'LineStyle','none');
    end
end

% outer frame
if (strcmp(frame,'yes'))
    patch( [tmin tmax tmax tmin], [fmin fmin fmax fmax], [-2 -2 -2 -2],[1 1 1]);
end