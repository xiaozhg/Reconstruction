function make_many_SASE_phase_3_22
clear all
tic
Nruns=20000;

npart   = 512;%128;							% n-macro-particles per bucket ?
s_steps = 100;%500;%600;	s/z=9/4					% n-sample points along bunch length
z_steps = 70;%350;%250;							% n-sample points along unulator
energy  = 4313.34;%3500;                         % electron energy [MeV]
eSpread = 1.0e-4;						% relative rms energy spread [ ]
emitN   = 1.2e-6;%0.5e-6;						% normalized transverse emittance [mm-mrad]
currentMax = 3400;%1000.0;					% peak current [Ampere]
beta = 26;%10.0;							% mean beta [meter]
unduPeriod = 0.03;%0.03						% undulator period [meter]
unduK = 3.5;							% undulator parameter, K [ ]
unduL = 30;%22.5;								% length of undulator [meter]
radWavelength = 1.5e-9;%22.78e-10;				% seed wavelength? [meter], used only in single-freuqency runs
dEdz = 0;							    % rate of relative energy gain or taper [keV/m], optimal~130
iopt = 5;								% 5=SASE, 4=seeded
P0 = 1000*0.0;							    % small seed input power [W]
constseed = 0;

inp_struc.npart			= npart;
inp_struc.s_steps		= s_steps;
inp_struc.z_steps		= z_steps;
inp_struc.energy		= energy;
inp_struc.eSpread		= eSpread;
inp_struc.emitN			= emitN;
inp_struc.currentMax	= currentMax;
inp_struc.beta			= beta;
inp_struc.unduPeriod	= unduPeriod;
inp_struc.unduK			= unduK;
inp_struc.unduL			= unduL;
inp_struc.radWavelength	= radWavelength;
inp_struc.dEdz			= dEdz;
inp_struc.iopt			= iopt;
inp_struc.P0			= P0;
inp_struc.constseed     = constseed;

mypath='/Users/xiao/Desktop/';
%mypath='/Users/dratner/Desktop/Data/GhostImaging/SASE_sim/12fs/';
%mypath='/Users/dratner/Dropbox/ResearchTopics/GhostImaging/Data/sase_sim/12fs/';
run_name=[mypath 'SASE_12fs_2p3nm_phase_' num2str(Nruns) 'runs.mat'];

tic
for j=1:Nruns
    [sfull,power_full,Pspec_j,phase_full,field_full,resWavelength,rho,z,power_z] = run_1_SASE(inp_struc);
    
    if j==1
        t=sfull*1e-6*1e15/3e8;
        power_tot=zeros(Nruns,length(power_full));
        Pspec_tot=zeros(Nruns,length(Pspec_j));        
        phase_tot=zeros(Nruns,length(phase_full));
        field_tot=zeros(Nruns,length(field_full));
    end    
    
    power_tot(j,:)=power_full;
    Pspec_tot(j,:)=Pspec_j;
    phase_tot(j,:)=phase_full;    
    field_tot(j,:)=field_full;
    
    if mod(j,10)==0
        disp([num2str(j) ' runs complete, ' num2str(toc) ' sec'])
    end
    if mod(j,500)==0
        power=power_tot(1:j,:);
        Pspec=Pspec_tot(1:j,:);
        phase=phase_tot(1:j,:);
        field=field_tot(1:j,:);
        save(run_name,'t','power','Pspec','phase','field','resWavelength')
    end
    
end

power=power_tot;
Pspec=Pspec_tot;
phase=phase_tot;
field=field_tot;
T=t(end);
sigt=sqrt(pi)*sqrt((2*pi*unduL)/(3*sqrt(3)*rho*unduPeriod))*resWavelength/(2.99792458E8*2*pi)*1e15;
N_t=size(t,2);
figure;plot(z,log(power_z))
xlabel('z/m')
ylabel('log(power\_z)')
save(run_name,'t','power','Pspec','phase','field','resWavelength','T','sigt','N_t')

function [sfull,power_full,Pspec_full,phase_full,field_full,resWavelength,rho,z,power_z] = run_1_SASE(inp_struc)

[z,power_z,s,power_s,rho,detune,field,field_s,gainLength,resWavelength] = sase1d(inp_struc);
% figure(1); semilogy(z,power_z*1E9,'r-')
% axis([0 ceil(max(z)) 1E9*power_z(2)/1.2 1E9*ceil(2*max(power_z))])
xlabel('{\itz} (m)')
ylabel('\langle{\itP}\rangle (W)')
title(['{\it\rho} = ' sprintf('%4.2e',rho) ',  {\itL_{G0}} = ' sprintf('%4.2f m',gainLength/sqrt(3)) ',  {\it\lambda_r} = ' sprintf('%5.3e m',resWavelength)])
enhance_plot('times',10,2,2)


fieldFFT = fftshift(fft(field.'));                  % unconjugate complex transpose by .'
Pspec=fieldFFT.*conj(fieldFFT);
% figure(32)
% plot(detune*2*rho,Pspec)
% %axis([-0.01 +0.01 0 1.2*max(Pspec)])
% xlabel('{(\Delta\omega)/\omega_r}')
% ylabel('{output spectral power} (a.u.)')
% enhance_plot('times',20,1,1)

field_s=field_s.';
field_s=field_s(2:end,1:end-1);

power_s_end=power_s(end,:);
field_s_end=field_s(end,:);

ds=s(2)-s(1);
power_slip = flipud(power_s(:,end)).';  % radiation that slipped off the beam
field_slip = flipud(field_s(:,end)).';  % radiation that slipped off the beam
power_full=cat(2,power_s_end,power_slip);
field_full=cat(2,field_s_end,field_slip);

field_fullFFT = fftshift(fft(field_full));                  % unconjugate complex transpose by .'
Pspec_full=field_fullFFT.*conj(field_fullFFT);
phase_full=angle(field_full);

ds=s(2)-s(1);
Np=length(power_full);
sfull=ds:ds:(Np)*ds;

% save(inp_struc.run_name,'s','power_s_end','Pspec','sfull','power_full')

% NEEDT TO CHECK THAT SLIPPAGE PER STEP IS 'ds'!!!!
um2fs=1e-6*1e15/3e8;
figure(2); plot(sfull*um2fs,power_full,s*um2fs,power_s_end)
toc



