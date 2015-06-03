function N = interactive_fourier_fit(time,data)
%% N = interactive_fourier_fit(time,data)
% Decides, together with the user, how many Fourier coefficients N are
% a good approximation for the signal (time,data)
% and returns this number

if (length(time)~=length(data))
    error('time and data must be the same length')
end
fprintf('Interactively determining Fourier fit for a signal with %i datapoints\n',length(time))



%% decide nfft for alpha interactively
not_satisfied = 1;
figure(55)
N=5;
usage = 'arrow key right/left: more/less Fourier modes +/- key: +/- 5 modes Esc key: satisfied with the result';
h=msgbox(usage);
uiwait(h)

time_fit = time(1):1e-3:time(end);
data_fit = zeros(size(time_fit));

while not_satisfied
    % make a least squares fit with the given nfft
    [ai,bi] = Fseries (time, data, N);   
    % evaluate it for plottingh
    for it = 1:length(time_fit)
        data_fit(it)   = Fseriesval( ai,bi, time_fit(it) );
    end
        
    % show the user what we acchieve iwth the current number of Fourier
    % modes
    plot( time, data, 'k', time_fit, data_fit, 'r'  )
    legend('input data', sprintf('Fourier fit with n=%i',N) )
    title(usage);
    
    % read input (from keyboard as well!)
    [x_tmp,y_tmp,button] = ginput(1);
    
    % decide what to do: more/less modes, or we're done?
    if (button == 27) % ESC finnishes
        h=msgbox('satisfied, bye bye');
        uiwait(h)
        not_satisfied = 0;
    elseif (button==28) % this is the left arrow key
        N = N-1;
    elseif (button==29) % this is the right arrow key
        N = N+1;
    elseif (button==43) % this is the + key
        N = N+5;
    elseif (button==45) % this is the - key
        N = N-5;
    end
end

fprintf('Result of interactive Fourier fit is %i modes\n',N)

end