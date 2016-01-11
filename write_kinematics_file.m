function write_kinematics_file(file, kinematics, header)
%% function write_kinematics_file(file, kinematics)
% write the Fourier series kinematics data specified in the struct
% kinematics to a FLUSI-compatible *.ini file
%--------------------------------------------------------------------------
% Input:
%       file - the kinematics.ini to write to
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

% This stops even when the file ends with .ini, at least in some versions of Matlab
%if (strfind(file,'.in'))
%    error('File ending not recongnized: did you forget the ending *.ini?')
%end

if ( strfind(file,'.ini') )
    %% Save to text file using new free-form *.ini style format 
    fid = fopen(file,'w');
    fprintf(fid,'; %s\n', header);
    fprintf(fid,'[kinematics]\n');
    fprintf(fid,'; if the format changes in the future\n');
    fprintf(fid,'format=2015-10-09; currently unused\n');
    fprintf(fid,'convention=flusi; currently unused\n');    
    fprintf(fid,'; what units, radiant or degree?\n');
    fprintf(fid,'units=degree;\n');
    fprintf(fid,'; is this hermite or Fourier coefficients?\n');
    fprintf(fid,'type=fourier;\n'); 
    fprintf(fid,';------------------------\n');
    fprintf(fid,'nfft_phi=%i;\n',  kinematics.nfft_phi  );
    fprintf(fid,'nfft_alpha=%i;\n',kinematics.nfft_alpha);
    fprintf(fid,'nfft_theta=%i;\n',kinematics.nfft_theta);
    fprintf(fid,';------------------------\n');
    fprintf(fid,'a0_phi=%16.10f;\n',kinematics.a0_phi);
    fprintf(fid,'ai_phi=');
    fprintf(fid,'%16.10f ',kinematics.ai_phi);
    fprintf(fid,';\n');
    fprintf(fid,'bi_phi=');
    fprintf(fid,'%16.10f ',kinematics.bi_phi);
    fprintf(fid,';\n');
    fprintf(fid,';------------------------\n');
    fprintf(fid,'a0_alpha=%16.10f;\n',kinematics.a0_alpha);
    fprintf(fid,'ai_alpha=');
    fprintf(fid,'%16.10f ',kinematics.ai_alpha);
    fprintf(fid,';\n');
    fprintf(fid,'bi_alpha=');
    fprintf(fid,'%16.10f ',kinematics.bi_alpha);
    fprintf(fid,';\n');
    fprintf(fid,';------------------------\n');
    fprintf(fid,'a0_theta=%16.10f;\n',kinematics.a0_theta);
    fprintf(fid,'ai_theta=');
    fprintf(fid,'%16.10f ',kinematics.ai_theta);
    fprintf(fid,';\n');
    fprintf(fid,'bi_theta=');
    fprintf(fid,'%16.10f ',kinematics.bi_theta);
    fprintf(fid,';\n');
    fclose(fid);
else
    %% Save to text file using old fixed-form *.in format
    warning('Using old fixed-form *.in kinematics file is deprecated, better use *.ini instead!')
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


end
