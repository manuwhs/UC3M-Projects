supportedidxs = find(support);
[supr,supc] = find(support);
rmin=min(supr); rmax=max(supr); cmin=min(supc); cmax=max(supc);
support_small = support(rmin:rmax, cmin:cmax);

for i=1:length(directory_list)
  fprintf('\nsupporting %s', ['../unpadded/' directory_list(i).name]);
  I = pgmRead(['../unpadded/' directory_list(i).name]);

  Io = I(rmin:rmax, cmin:cmax) .* support_small;

  pgmWrite(Io, ['../supported/' directory_list(i).name], [0 255]);
end
