function kinematics = read_kinematics_file(file)
%% function kinematics = read_kinematics_file(file)
% returns a struct with the fourier coefficients read from a flusi
% kinematics.in file
%--------------------------------------------------------------------------
% Input:
%       file - the kinematics.in to visualize
%--------------------------------------------------------------------------
% Output:
%       kinematics: struct with the following components:
%
%       kinematics.nfft_phi
%       kinematics.nfft_alpha
%       kinematics.nfft_theta
%       kinematics.a0_phi
%       kinematics.ai_phi
%       kinematics.bi_phi
%       kinematics.a0_alpha
%       kinematics.ai_alpha
%       kinematics.bi_alpha
%       kinematics.a0_theta
%       kinematics.ai_theta
%       kinematics.bi_theta
%
%--------------------------------------------------------------------------

if (strfind(file,'.ini'))
    %% read file from new free-form *.ini file
    kinematics.nfft_phi   = read_number_inifile(file,'kinematics','nfft_phi');
    kinematics.nfft_alpha = read_number_inifile(file,'kinematics','nfft_alpha');
    kinematics.nfft_theta = read_number_inifile(file,'kinematics','nfft_theta');
    
    kinematics.a0_phi = read_number_inifile(file,'kinematics','a0_phi');
    kinematics.ai_phi(:,1) = read_number_inifile(file,'kinematics','ai_phi');
    kinematics.bi_phi(:,1) = read_number_inifile(file,'kinematics','bi_phi');
    
    kinematics.a0_alpha = read_number_inifile(file,'kinematics','a0_alpha');
    kinematics.ai_alpha(:,1) = read_number_inifile(file,'kinematics','ai_alpha');
    kinematics.bi_alpha(:,1) = read_number_inifile(file,'kinematics','bi_alpha');
    
    kinematics.a0_theta = read_number_inifile(file,'kinematics','a0_theta');
    kinematics.ai_theta(:,1) = read_number_inifile(file,'kinematics','ai_theta');
    kinematics.bi_theta(:,1) = read_number_inifile(file,'kinematics','bi_theta');
else
    %% read from old deprecated fixed-form *.in file
    warning('Using old fixed-form *.in kinematics file is deprecated, better use *.ini instead!')
    fid = fopen(file,'r');
    % skip header from the file
    dummy = fgetl(fid);
    kinematics.nfft_phi   = sscanf(fgetl(fid),'%i');
    kinematics.nfft_alpha = sscanf(fgetl(fid),'%i');
    kinematics.nfft_theta = sscanf(fgetl(fid),'%i');
    
    % read values for phi from the file
    dummy = fgetl(fid);
    dummy = fgetl(fid);
    kinematics.a0_phi = sscanf(fgetl(fid),'%f');
    dummy = fgetl(fid);
    kinematics.ai_phi = sscanf(fgetl(fid),'%f',kinematics.nfft_phi);
    dummy = fgetl(fid);
    kinematics.bi_phi = sscanf(fgetl(fid),'%f',kinematics.nfft_phi);
    
    % read values for alpha from the file
    dummy = fgetl(fid);
    dummy = fgetl(fid);
    kinematics.a0_alpha = sscanf(fgetl(fid),'%f');
    dummy = fgetl(fid);
    kinematics.ai_alpha = sscanf(fgetl(fid),'%f',kinematics.nfft_alpha);
    dummy = fgetl(fid);
    kinematics.bi_alpha = sscanf(fgetl(fid),'%f',kinematics.nfft_alpha);
    
    % read values for theta from the file
    dummy = fgetl(fid);
    dummy = fgetl(fid);
    kinematics.a0_theta = sscanf(fgetl(fid),'%f');
    dummy = fgetl(fid);
    kinematics.ai_theta = sscanf(fgetl(fid),'%f',kinematics.nfft_theta);
    dummy = fgetl(fid);
    kinematics.bi_theta = sscanf(fgetl(fid),'%f',kinematics.nfft_theta);
    
    % we're done with the file
    fclose(fid);
end

end