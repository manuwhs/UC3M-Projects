eigenvalues = diag(s_faces);
save ../eigenvalues.dat eigenvalues -ascii;

supportedidxs = find(support_small);
I = zeros(size(support_small));
I(supportedidxs) = B_mean;
pgmWrite(I, '../eigfaces-for-your-viewing-pleasure/average.pgm');
pgmWrite(support_small, '../eigfaces/support.pgm');
fid = fopen('../eigfaces/average.dat', 'wb');
fwrite(fid, I', 'float32');
fclose(fid);

for i=1:size(B,2)
  I(supportedidxs) = u_faces(:,i);

  fprintf('\nwriting ../eigfaces/%03d.dat', i);
  pgmWrite(I, sprintf('../eigfaces-for-your-viewing-pleasure/%03d.pgm', i));

  fid = fopen(sprintf('../eigfaces/%03d.dat', i), 'wb');
  fwrite(fid, I', 'float32');
  fclose(fid);
end
