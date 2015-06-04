function visualize_kinematics_file( file )
%% function visualize_kinematics_file( file )
% Read in a FLUSI Fourier kinematics file (typically *.in), and plot the 
% kinematics, saving them to an EPS graphics file
%--------------------------------------------------------------------------
% Input:
%       file - the kinematics.in to visualize
%--------------------------------------------------------------------------
% Output:
%       writes kinematics.in.eps directly to disk.
%--------------------------------------------------------------------------


% read in kinematics.in file:
kine = read_kinematics_file(file);



%% construct angles from Fourier series
time = 0:1e-3:1;
[phi,alpha,theta] = evaluate_kinematics_file_time(time,kine);



%% PLOT
set(0,'DefaultaxesFontSize',9);
set(0,'DefaulttextFontsize',9);
set(0,'DefaultaxesFontName','Times');
set(0,'DefaulttextFontName','Times');

% Create figure
h_fig = figure;
set(h_fig,'Units','centimeters','Position',0.65*[10 8.0 14.0 8.8],'Resize','on','PaperPositionMode','auto'); 
clf;


plot( time, phi, time, alpha, time, theta, 'LineWidth', 1 )


% Annotate
axis([0 1 -110 110]);
set(gca,'XTick',[0:0.2:1]);
set(gca,'YTick',[-90:30:90]);
legend('positional','feathering','elevation','Location','SouthWest');
xlabel('wingbeat time fraction');
ylabel('angle, deg');


% Print figure
print('-depsc', [file '.eps']);

end




function y = Fseriesval(a,b,t)
    N = length(b);
    ai = a(2:end);
    bi = b;

    % constant
    y = a(1)/2; 
    for k = 1:N
     y = y + ai(k)*cos(2*pi*k*t) + bi(k)*sin(2*pi*k*t) ;
    end
end