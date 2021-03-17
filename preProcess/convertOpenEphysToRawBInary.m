function ops = convertOpenEphysToRawBInary(ops)

% input .continuous files is in data base temp
% output .bin file in the same folder.

fname       = ops.fbinary;
UTmkdir(fname);

fidout      = fopen(fname, 'w');
if(fidout==-1)
    error(['Could not open file: ',fname])
end



% generates a cell (channel) of structs (blocks) of .continuous files across
% all different directories corresponding to different experimental blocks
ch=load(ops.chanMap);
chans=sort(ch.chanMap);
fs=cell(ops.NchanTOT,1);
for j = 1:ops.NchanTOT
    ops.chanMap_KiloRaw(ch.chanMap==chans(j))=j;
end

for j = 1:ops.NchanTOT
    ops.chanMap_KiloRaw(ch.chanMap==chans(j))=j;
    for k = 1:length(ops.root)
        sa(k,1) = dir(fullfile(ops.root{k}, sprintf('*CH%d.continuous', chans(j)) ));
    end
    fs{j} = sa;
end
nblocks = cellfun(@(x) numel(x), fs);
if numel(unique(nblocks))>1
   error('different number of blocks for different channels!') 
end
%
nBlocks     = unique(nblocks);
nSamples    = 1024;  % fixed to 1024 for now!

fid = cell(ops.NchanTOT, 1);

fprintf('Concatenating Open-Ephys data to a single binary file.\n')
tic
ops.nSamplesBlocks=nan(1,nBlocks);
for k = 1:nBlocks
    fprintf('block %d of %d \n', k, nBlocks')
    for j = 1:ops.NchanTOT
        fid{j} = fopen(fullfile(fs{j}(k).folder, fs{j}(k).name));
        % discard header information
        fseek(fid{j}, 1024, 0);
    end
    %
    nsamps = 0;
    flag = 1;
    while 1
        samples = zeros(nSamples * 1000, ops.NchanTOT, 'int16');
        for j = 1:ops.NchanTOT
            
            collectSamps    = zeros(nSamples * 1000, 1, 'int16');
            rawData         = fread(fid{j}, 1000 * (nSamples + 6), '1030*int16', 10, 'b');
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
        if(isfield(ops,'common_rejection_mode'))
                switch ops.common_rejection_mode
                    case 'none'
                    case 'mean'
                        samples=samples-repmat(mean(samples),size(samples,1),1);
                    case 'median'
                        samples=samples-repmat(median(samples),size(samples,1),1);
                    otherwise
                        error(['Unknown comon rejection mode ',ops.common_rejection_mode])
                end
        end
            
        fwrite(fidout, samples, 'int16');
        
        nsamps = nsamps + size(samples,2);
        
        if flag==0
            break;
        end
    end
    ops.nSamplesBlocks(k) = nsamps;
    
    for j = 1:ops.NchanTOT
       fclose(fid{j}); 
    end
    
end
    
fclose(fidout);
fprintf('Done concatenating, ')
toc
fprintf('\n')

