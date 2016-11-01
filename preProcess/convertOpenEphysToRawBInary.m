function ops = convertOpenEphysToRawBInary(ops)

%fname       = fullfile(ops.root, sprintf('%s.dat', ops.fbinary)); 
fname       = ops.fbinary; 
fidout      = fopen(fname, 'w');
UTmkdir(fname);
if(fidout==-1)
    error(['Could not open file: ',fname])
end
%
fs=cell(ops.Nchan,1);
for j = 1:ops.Nchan
   for k=1:length(ops.root)
       d=dir(fullfile(ops.root{k}, sprintf('*CH%d.continuous', j) ));
       [d.dir]=deal(ops.root{k});
       fs{j} = [fs{j} d];
   end
end
nblocks = cellfun(@(x) numel(x), fs);
if numel(unique(nblocks))>1
   error('different number of blocks for different channels!') 
end
%
nBlocks     = unique(nblocks);
nSamples    = 1024;  % fixed to 1024 for now!

fid = cell(ops.Nchan, 1);
fprintf('Concatenating Open-Ephys data to a single binary file.')
tic
for k = 1:nBlocks
    for j = 1:ops.Nchan
        fid{j}             = fopen(fullfile(fs{j}(k).dir, fs{j}(k).name));
        % discard header information
        fseek(fid{j}, 1024, 0);
    end
    %
    nsamps = 0;
    flag = 1;
    while 1
        samples = zeros(nSamples * 1000, ops.Nchan, 'int16');
        for j = 1:ops.Nchan
            collectSamps    = zeros(nSamples * 1000, 1, 'int16');
            
            rawData         = fread(fid{j}, 1000 * (nSamples + 6), '1030*int16', 10, 'b');

            %nbatches        = ceil(numel(rawData)/(nSamples+6));
            nbatches        = floor(numel(rawData)/(nSamples+6));
            for s = 1:nbatches
                rawSamps = rawData((s-1) * (nSamples + 6) +6+ [1:nSamples]);
                collectSamps((s-1)*nSamples + [1:nSamples]) = rawSamps;
            end
            samples(:,j)         = collectSamps;
        end
        
        if nbatches<1000
            flag = 0;
        end
        if flag==0
            samples = samples(1:s*nSamples, :);
        end
       
        samples         = samples';
        fwrite(fidout, samples, 'int16');
        
        nsamps = nsamps + size(samples,2);
        
        if flag==0
            break;
        end
    end
    ops.nSamplesBlocks(k) = nsamps;
    
    for j = 1:ops.Nchan
       fclose(fid{j}); 
    end
    
end
    
fclose(fidout);

toc