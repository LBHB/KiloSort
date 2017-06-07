function ops = convertMANTAToRawBinary(ops,do_write)

if(nargin<2)
    do_write=true;
end
%fname       = fullfile(ops.root, sprintf('%s.dat', ops.fbinary));
fname       = ops.fbinary;
UTmkdir(fname);
if(do_write)
    fidout      = fopen(fname, 'w');
    if(fidout==-1)
        error(['Could not open file: ',fname])
    end
end
%
nFiles=length(ops.root);
fprintf('Concatenating MANTA data to a single binary file.')
tic
for iFile=1:nFiles
    evpfile=ops.globalparams(iFile).evpfilename;
    [pp,bb,ee]=fileparts(evpfile);
    bbpref=strsep(bb,'_');
    bbpref=bbpref{1};
    checktgzevp=[pp filesep 'raw' filesep bbpref '.tgz'];
    if exist(checktgzevp,'file'),
        evpfile=checktgzevp;
    end
    rawData=evpread(evpfile,'spikeelecs',1:ops.Nchan,'globalparams',ops.globalparams(iFile),'runinfo',ops.runinfo(iFile),'filterstyle','none');
    samples         = int16(rawData.Spike'*1e6*ops.raw_scale_factor);
    ops.trial_onsets_{iFile}=rawData.STrialidx';
    if(iFile==1)
        ops.fs=rawData.Info.SR;
    elseif ops.fs~=rawData.Info.SR
        error('Sampling rate was inconsistent across files!')
    end
    if(do_write)
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
        written_count=fwrite(fidout, samples, 'int16');
    end
    ops.nSamplesBlocks(iFile) = size(samples,2); %Blocks means files :-/
end
if(do_write)
    fclose(fidout);
end
toc