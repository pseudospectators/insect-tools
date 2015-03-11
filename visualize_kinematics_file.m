function visualize_kinematics_file( file )
%% function visualize_kinematics_file( file )
% Read in a FLUSI Fourier kinematics file (typically *.in), and plot the 
% kinematics, saving them to eps
%--------------------------------------------------------------------------
% Input:
%       file - the kinematics.in to visualize
%--------------------------------------------------------------------------
% Output:
%       writes kinematics.in.eps directly to disk.
%--------------------------------------------------------------------------


%% read in *.in file
fid = fopen(file,'r');
% skip header from the file
dummy = fgetl(fid);
nfft_phi   = sscanf(fgetl(fid),'%i');
nfft_alpha = sscanf(fgetl(fid),'%i');
nfft_theta = sscanf(fgetl(fid),'%i');

% read values for phi from the file
dummy = fgetl(fid);
dummy = fgetl(fid);
a0_phi = sscanf(fgetl(fid),'%f');
dummy = fgetl(fid);
ai_phi = sscanf(fgetl(fid),'%f',nfft_phi);
dummy = fgetl(fid);
bi_phi = sscanf(fgetl(fid),'%f',nfft_phi);

% read values for alpha from the file
dummy = fgetl(fid);
dummy = fgetl(fid);
a0_alpha = sscanf(fgetl(fid),'%f');
dummy = fgetl(fid);
ai_alpha = sscanf(fgetl(fid),'%f',nfft_alpha);
dummy = fgetl(fid);
bi_alpha = sscanf(fgetl(fid),'%f',nfft_alpha);

% read values for theta from the file
dummy = fgetl(fid);
dummy = fgetl(fid);
a0_theta = sscanf(fgetl(fid),'%f');
dummy = fgetl(fid);
ai_theta = sscanf(fgetl(fid),'%f',nfft_theta);
dummy = fgetl(fid);
bi_theta = sscanf(fgetl(fid),'%f',nfft_theta);

% we're done with the file
fclose(fid);



%% construct angles from Fourier series
time = 0:1e-3:1;

phi   = zeros(size(time));
alpha = zeros(size(time));
theta = zeros(size(time));

for it = 1:length(time)
   phi(it)   = Fseriesval( [a0_phi; ai_phi], bi_phi, time(it) );
   alpha(it) = Fseriesval( [a0_alpha; ai_alpha], bi_alpha, time(it) );
   theta(it) = Fseriesval( [a0_theta; ai_theta], bi_theta, time(it) );
end



%% PLOT
set(0,'DefaultaxesFontSize',9);
set(0,'DefaulttextFontsize',9);
set(0,'DefaultaxesFontName','Times');
set(0,'DefaulttextFontName','Times');

% Create figure
h_fig = figure(1);
set(h_fig,'Units','centimeters','Position',0.65*[10 8.0 14.0 8.8],'Resize','on','PaperPositionMode','auto'); 
clf;


plot( time, phi, time, alpha, time, theta, 'LineWidth', 1 )


% Annotate
axis([0 1 -90 90]);
set(gca,'XTick',[0:0.2:1]);
set(gca,'YTick',[-90:30:90]);
legend('positional','feathering','elevation','Location','SouthEast');
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