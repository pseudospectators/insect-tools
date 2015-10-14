function value = read_number_inifile( file, section, keyword)
%% function value = read_number_inifile( file, section, keyword)
% from the specified ini file "file"
% find the section "section" (e.g. "[kinematics]" ) 
% and the keyword, e.g. variable=0.0202;
% and return the value, so 0.0202


value = str2num( read_str_inifile( file, section, keyword) );

end