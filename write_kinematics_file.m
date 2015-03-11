function write_kinematics_file(file, kinematics, header)
%% function write_kinematics_file(file, kinematics)
% write the Fourier series kinematics data specified in the struct
% kinematics to a FLUSI-compatible *.in file
%--------------------------------------------------------------------------
% Input:
%       file - the kinematics.in to write to
%       header - header line for file
%
%       kinematics: struct with the following components:
%           kinematics.nfft_phi
%           kinematics.nfft_alpha
%           kinematics.nfft_theta
%           kinematics.a0_phi
%           kinematics.ai_phi
%           kinematics.bi_phi
%           kinematics.a0_alpha
%           kinematics.ai_alpha
%           kinematics.bi_alpha
%           kinematics.a0_theta
%           kinematics.ai_theta
%           kinematics.bi_theta
%       
%--------------------------------------------------------------------------


%% Save to text file
fid = fopen(file,'w');
fprintf(fid,'%% %s\n', header);
fprintf(fid,'%i %% nfft_phi\n',  kinematics.nfft_phi  );
fprintf(fid,'%i %% nfft_alpha\n',kinematics.nfft_alpha);
fprintf(fid,'%i %% nfft_theta\n',kinematics.nfft_theta);
fprintf(fid,'%%------------------------\n');
fprintf(fid,'%% a0 for phi\n');
fprintf(fid,'%-16.10f',kinematics.a0_phi);
fprintf(fid,'\n');
fprintf(fid,'%% ai for phi\n');
fprintf(fid,'%-16.10f',kinematics.ai_phi);
fprintf(fid,'\n');
fprintf(fid,'%% bi for phi\n');
fprintf(fid,'%-16.10f',kinematics.bi_phi);
fprintf(fid,'\n');
fprintf(fid,'%%------------------------\n');
fprintf(fid,'%% a0 for alpha\n');
fprintf(fid,'%-16.10f',kinematics.a0_alpha);
fprintf(fid,'\n');
fprintf(fid,'%% ai for alpha\n');
fprintf(fid,'%-16.10f',kinematics.ai_alpha);
fprintf(fid,'\n');
fprintf(fid,'%% bi for alpha\n');
fprintf(fid,'%-16.10f',kinematics.bi_alpha);
fprintf(fid,'\n');
fprintf(fid,'%%------------------------\n');
fprintf(fid,'%% a0 for theta\n');
fprintf(fid,'%-16.10f',kinematics.a0_theta);
fprintf(fid,'\n');
fprintf(fid,'%% ai for theta\n');
fprintf(fid,'%-16.10f',kinematics.ai_theta);
fprintf(fid,'\n');
fprintf(fid,'%% bi for theta\n');
fprintf(fid,'%-16.10f',kinematics.bi_theta);
fprintf(fid,'\n');
fclose(fid);
end