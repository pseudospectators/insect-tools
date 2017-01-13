function value = read_number_inifile( file, section, keyword)
%% function value = read_number_inifile( file, section, keyword)
% from the specified ini file "file"
% find the section "section" (e.g. "[kinematics]" )
% and the keyword, e.g. variable=0.0202;
% and return the value, so 0.0202

string = read_str_inifile( file, section, keyword);

if (isempty(string))
    value= 0;
else    
    value = str2num( string );
end

end