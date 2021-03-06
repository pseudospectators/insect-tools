function visualize_kinematics_file( file, invert_theta )
%% function visualize_kinematics_file( file, invert_theta )
% Read in a FLUSI Fourier kinematics file (typically *.ini), and plot the 
% kinematics, saving them to an EPS graphics file
%--------------------------------------------------------------------------
% Input:
%       file - the kinematics.in to visualize
%       invert_theta (0/1), optional
%--------------------------------------------------------------------------
% Output:
%       writes kinematics.in.eps directly to disk.
%--------------------------------------------------------------------------

if nargin==1
    invert_theta=0;
end

% read in kinematics.ini file:
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
% set(h_fig,'Units','centimeters','Position',0.65*[10 8.0 19.0 8.8],'Resize','on','PaperPositionMode','auto'); 
set(h_fig,'Units','centimeters','Position',0.65*[10 8.0 14.0 8.8],'Resize','on','PaperPositionMode','auto'); 
clf;

if invert_theta
    plot( time, phi, time, alpha, time, -theta, 'LineWidth', 1 )
else
    plot( time, phi, time, alpha, time, theta, 'LineWidth', 1 )
end


% Annotate
a = axis;
% factor = 1.2;
% axis([0 1 -factor*a(4) factor*a(4)]);
set(gca,'XTick',[0:0.2:1]);
set(gca,'YTick',[-90:30:90]);
legend('positional','feathering','elevation','Location','NorthEastOutside');
legend('positional','feathering','elevation','Location','Best');
% legend('positional','feathering','elevation');
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