Io = ones(320,320)*255;
top_pad = round((320-243)/2);

for i=1:length(directory_list)
  fprintf('\npadding yalefaces/%s', directory_list(i).name);
  I=pgmRead(['../faces/' directory_list(i).name]);

  Io(top_pad+(0:size(I,1)-1), 1:size(I,2)) = I;

  pgmWrite(Io, ['../padded/' directory_list(i).name]);
end
