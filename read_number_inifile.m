function value = read_number_inifile( file, section, keyword)
%% function value = read_number_inifile( file, section, keyword)
% from the specified ini file "file"
% find the section "section" (e.g. "[kinematics]" )
% and the keyword, e.g. variable=0.0202;
% and return the value, so 0.0202

string = read_str_inifile( file, section, keyword);

% newer ini files may contain vectors in an arbitrary notation (ie
% matrices), remove the keywords for that:
string = strrep( string, '(/', '' );
string = strrep( string, '/)', '' );

if (isempty(string))
    value = 0;
else    
    value = str2num( string );
end

   saveas(gca,[file '.eps'],'epsc');
end