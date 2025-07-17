% 1D SW equations in periodic deomain in physical formulation
% not in conservative form
% surface is z1, total depth H
% including bottom and surface stress
% mvindex=1;
% % this is the name of the resulting movie file
% mvfile='arcticseiche1';
close all
clear all

drawpics=1;
compute_spec=0;
save_files=0;
% physical parameters
g=9.81*0.01; f0=1e-4; c_d=0.00025e0; cw=0.0001; rhoa=1.2; rhow=1023; denfact=rhoa/rhow; cw=cw*denfact;
% number of grid points (total is N+1) and the grid
H=10; H02o6=H*H/6;H0=H;

L=1e4; c0=sqrt(g*H);bigt=L/c0;

N=8192; 
z=linspace(-1,1,N+1);
z=z(1:end-1);
xf=z*L; dx=xf(2)-xf(1); x=xf((N/2+1):end);
dk=pi/L;
ksvec=zeros(size(x));
ksvec(1)=0; ksvec(N/2+1)=0;
for ii=2:(N/2)
   ksvec(ii)=ii-1;
   ksvec(N/2+ii)=-N/2 + ii -1;
end
k=ksvec*dk; k2=k.*k; ik=sqrt(-1)*k;

%Build our `radial' filter
filtorder=8;
kmax=max(k(:));
cutoff=0.65;
kcrit=kmax*cutoff;
alpha1=2;
alpha2=1; 
kmag=sqrt(k.*k);
myfilter=exp(-alpha1*(kmag/(alpha2*kcrit)).^filtorder);


cfl=dx/c0;
numsteps=6000;
numouts=10;
dt=2.0; %dt=bigt*1e-3;
t=0;
NL=1; ROT=1; f=ROT*f0; NH=1;

nhfact=1./(1+NH*k2*H*H/6);
%nhfact=1;

um1=zeros(size(xf));
vm1=zeros(size(xf));
%zm1=H*(1-0.1*0.5*(tanh((xf-1e4)/1e3)-tanh((xf+1e4)/1e3)));
zm1=H*(1-0.05*cos(2*pi*xf/L));
% for ii=1:2,
%  zm1=zm1+0.05*H*cos((3+4*ii)*pi*xf/L);
% end

um2=um1;
vm2=vm1;
zm2=zm1;

um3=um1;
vm3=vm1;
zm3=zm1;

um4=um1;
vm4=vm1;
zm4=zm1;
if save_files==1
    filenamestr='outinfo';
    save(filenamestr,'xf','g','H','c_d','f0','dt','zm1','um1','vm1')
end
if drawpics==1
    figure(2), clf
    set(gcf,'DefaultLineLineWidth',2,'DefaultTextFontSize',12,...
            'DefaultTextFontWeight','bold','DefaultAxesFontSize',12,...
              'DefaultAxesFontWeight','bold');


     subplot(2,1,1)
     plot(xf,(zm1-H)/H,'k-'),title(['t = ' num2str(t,2)])
     axis([0 L -0.1 0.1])
     subplot(2,1,2)
     plot(xf,um1/sqrt(g*H0),'k-',xf,vm1/sqrt(g*H0),'b-'); xlabel('x');
     axis([0 L -0.05 0.05])
     drawnow
end
 
 t=t+dt;
    uf1=fft(um1);zf1=fft(zm1);uhef1=fft(um1.*zm1);uuf1=0.5*fft(um1.^2); vf1=fft(vm1);
    spd=sqrt(um1.*um1+vm1.*vm1);
    drgxf=fft(-c_d.*um1.*spd);
    drgyf=fft(-c_d.*vm1.*spd);
    
    un1=um1+dt*(real(ifft((-i*NL*k.*uuf1-i*g*k.*zf1+drgxf+f*vf1))));
    vn1=vm1+dt*(um1.*real(ifft(-i*NL*k.*vf1))+drgyf-f*um1);
    zn1=zm1+dt*(real(ifft(-i*k.*uhef1)));
    
    uf2=fft(um2);zf2=fft(zm2);uhef2=fft(um2.*zm2);uuf2=0.5*fft(um2.^2); vf2=fft(vm2);
    
    un2=um2+dt*(real(ifft((-i*NL*k.*uuf2-i*g*k.*zf2+f*vf2))));
    vn2=vm2+dt*(um1.*real(ifft(-i*NL*k.*vf2))-f*um2);
    zn2=zm2+dt*(real(ifft(-i*k.*uhef2)));
    
    uf3=fft(um3);zf3=fft(zm3);uHf=fft(um3.*H);uuf3=0.5*fft(um3.^2); vf3=fft(vm3);
    spd=sqrt(um3.*um3+vm3.*vm3);
    drgxf=fft(-c_d.*um3.*spd);
    drgyf=fft(-c_d.*vm3.*spd);
    
    un3=um3+dt*(real(ifft((-i*g*k.*zf3+drgxf+f*vf3))));
    vn3=vm3+dt*real(ifft(drgyf-f*um3));
    zn3=zm3+dt*(real(ifft(-i*k.*uHf)));

    uf4=fft(um4);zf4=fft(zm4);uhef4=fft(um4.*zm4);uuf4=0.5*fft(um4.^2); vf4=fft(vm4);
    spd=sqrt(um4.*um4+vm4.*vm4);
    drgxf=fft(-c_d.*um4.*spd);
    drgyf=fft(-c_d.*vm4.*spd);
    
    un4=um4+dt*(real(ifft((-i*NL*k.*uuf4-i*g*k.*zf4+drgxf))));
    vn4=vm4+dt*(um4.*real(ifft(-i*NL*k.*vf4))+drgyf);
    zn4=zm4+dt*(real(ifft(-i*k.*uhef4)));
for ii=1:numouts
 for jj=1:numsteps
    fnow=0*fft(1e-2*sin(2*pi*t/(5*16.4873))*cos(pi*xf/L));
    t=t+dt;
    uf1=fft(un1);zf1=fft(zn1);uhef1=fft(un1.*(H+NL*(zn1-H)));uuf1=0.5*fft(un1.^2); vf1=fft(vn1);
    spd=sqrt(un1.*un1+vn1.*vn1);
    drgxf=fft(-c_d.*un1.*spd);
    drgyf=fft(-c_d.*vn1.*spd);
    
    
    up1=um1+2*dt*(real(ifft(nhfact.*(-i*NL*k.*uuf1-i*g*k.*zf1+fnow+drgxf+f*vf1))));
    vp1=vm1+2*dt*(un1.*real(ifft(-i*NL*k.*vf1))+drgyf-f*un1);
    zp1=zm1+2*dt*(real(ifft(-i*k.*uhef1)));
    
    um1=un1;vm1=vn1;zm1=zn1;
    un1=real(ifft(myfilter.*fft(up1)));
    vn1=real(ifft(myfilter.*fft(vp1)));
    zn1=real(ifft(myfilter.*fft(zp1)));
 
    uf2=fft(un2);zf2=fft(zn2);uhef2=fft(un2.*(H+NL*(zn2-H)));uuf2=0.5*fft(un2.^2); vf2=fft(vn2);
   
    up2=um2+2*dt*(real(ifft(nhfact.*(-i*NL*k.*uuf2-i*g*k.*zf2+fnow+f*vf2))));
    vp2=vm2+2*dt*(un2.*real(ifft(-i*NL*k.*vf2))-f*un2);
    zp2=zm2+2*dt*(real(ifft(-i*k.*uhef2)));
    
    um2=un2;vm2=vn2;zm2=zn2;
    un2=real(ifft(myfilter.*fft(up2)));
    vn2=real(ifft(myfilter.*fft(vp2)));
    zn2=real(ifft(myfilter.*fft(zp2)));

    uf3=fft(un3);zf3=fft(zn3);uH3=fft(un3.*H); vf3=fft(vn3);
    spd=sqrt(un3.*un3+vn3.*vn3);
    drgxf=fft(-c_d.*un3.*spd);
    drgyf=fft(-c_d.*vn3.*spd);
  
    up3=um3+2*dt*(real(ifft(nhfact.*(-i*g*k.*zf3+fnow+f*vf3+drgxf))));
    vp3=vm3+2*dt*(-f*un3+real(ifft(drgyf)));
    zp3=zm3+2*dt*(real(ifft(-i*k.*uH3)));
    
    um3=un3;vm3=vn3;zm3=zn3;
    un3=real(ifft(myfilter.*fft(up3)));
    vn3=real(ifft(myfilter.*fft(vp3)));
    zn3=real(ifft(myfilter.*fft(zp3)));

    uf4=fft(un4);zf4=fft(zn4);uhef4=fft(un4.*(H+NL*(zn4-H)));uuf4=0.5*fft(un4.^2); vf4=fft(vn4);
    spd=sqrt(un4.*un4+vn4.*vn4);
    drgxf=fft(-c_d.*un4.*spd);
    drgyf=fft(-c_d.*vn4.*spd);
    
    up4=um4+2*dt*(real(ifft(nhfact.*(-i*NL*k.*uuf4-i*g*k.*zf4+fnow+drgxf))));
    vp4=vm4+2*dt*(un4.*real(ifft(-i*NL*k.*vf4))+drgyf);
    zp4=zm4+2*dt*(real(ifft(-i*k.*uhef4)));
    
    um4=un4;vm4=vn4;zm4=zn4;
    un4=real(ifft(myfilter.*fft(up4)));
    vn4=real(ifft(myfilter.*fft(vp4)));
    zn4=real(ifft(myfilter.*fft(zp4)));
 end
if drawpics==1
      figure
      clf
      betterplots
      subplot(3,1,1)
      plot(xf/L,(zn1-H)/H,'k-',xf/L,(zn2-H)/H,'b-',xf/L,(zn3-H)/H,'g-',xf/L,(zn4-H)/H,'r-')
      axis([0 1 -0.075 0.075])
      grid on
      title(['t = ' num2str(t/bigt,2)])
      %title(['t = ' num2str(t/bigt,2) ' full (black), no drag (blue), linearized (red), no rotation (green)'])
      ylabel('scaled eta')
     % legend('all in','no drag','no NL','Location','SouthWest')
      subplot(3,1,2)
      plot(xf/L,un1/sqrt(g*H0),'k-',xf/L,un2/sqrt(g*H0),'b-',xf/L,un3/sqrt(g*H0),'g-',xf/L,un4/sqrt(g*H0),'r-'); 
      %legend('all in','no drag','no NL')
      axis([0 1 -0.075 0.075])
      grid on
      ylabel('scaled u')
      subplot(3,1,3)
      plot(xf/L,vn1/sqrt(g*H0),'k-',xf/L,vn2/sqrt(g*H0),'b-',xf/L,vn3/sqrt(g*H0),'g-',xf/L,vn4/sqrt(g*H0),'r-'); xlabel('x');
      legend('all in','no drag','linear','no f')
      axis([0 1 -0.075 0.075])
      grid on
      ylabel('scaled v')
    if compute_spec==1 
         myspec1=abs(fft(un1));
         myspec2=abs(fft(un2));
         myspec3=abs(fft(un3));
         kf=k(1:200)/(pi/L);

         % The 1e-11 added factor is need so Matlab's plotting doesn't mess up
         subplot(2,2,3)
         semilogy(kf,myspec1(1:200),'k^-',kf,myspec2(1:200),'bo-')
         axis([0 100 1e-6 5e2])
         xlabel('k')
         ylabel('Log PSD')
         subplot(2,2,4)
         semilogy(kf,myspec1(1:200),'k^-',kf,myspec3(1:200)+1e-16,'rs-')
         axis([0 100 1e-6 5e2])
         xlabel('k')
         ylabel('Log PSD')
    end
    %  subplot(4,1,4)
    %  semilogy(k(1:200),myspec1(1:200),'k-',k(1:200),myspec2(1:200),'b-',k(1:200),myspec3(1:200),'r-')
    %  axis([0 max(k)*0.2 1e-4 1e2])
    %  xlabel('k')
    %  ylabel('log PSD')
end
 drawnow
 if save_files==1
  filenamestr = ['out' sprintf('%07d',ii) '.mat'];
  save(filenamestr,'un1','un2','un3','vn1','vn2','vn3','zn1','zn2','zn3');
 end
end
