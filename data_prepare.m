% Prepare data for Python
clc
%*******************************************************************************
%                          Loading the data set
%*******************************************************************************

disp(' ')
disp(['Loading data...'])

data=load('birthdata.txt'); %data set for first time white mothers

N=size(data,1); %number of observations

byear=data(:,1); %birth year
fage=data(:,2); %father's age 
mage=data(:,3); %mother's age
feduc=data(:,4); %father's education
meduc=data(:,5); %mother's education 
terms=data(:,6); %number of previous terminated pregnancies 
gestation=data(:,7); %gestation length
prenatal=data(:,8); %month of first prenatal visit
prenatal_visits=data(:,9); %number of prenatal visits
mom_zip=data(:,10); %mom's zip code
wtgain=data(:,11); %mother's weight gain
anemia=data(:,12); %anemia
diabetes=data(:,13); %gestational diabetes
hyperpr=data(:,14); %high blood pressure
amnio=data(:,15); %amniocentesis was perfomed during pregnancy
ultra=data(:,16); %ultrasound exam was done
male=data(:,17); %baby's gender
feducmiss=data(:,18); %father's education missing
fagemiss=data(:,19); %father's age missing
married=data(:,20); %marital status
bweight=data(:,21); %birthweight in grams -- DEPENDENT VARIABLE
smoke=data(:,22); %smoking indicator -- TREATMENT DUMMY
drink=data(:,23); %alcohol use during pregnancy
kessner_ad=data(:,24); %adequate prenatal care by kessner index
kessner_inad=data(:,25); %inadequate prenatal care by kessner index
med_inc=data(:,26); %median income in mother's zip code
pc_inc=data(:,27); %per capita income in mother's zip code
long=data(:,28); %longitude of mother's zip code
lat=data(:,29); %latitude of mother's zip code
popdens=data(:,30); %population density in mother's zip code

prenatal(prenatal==0)=10; %month of first prenatal visit. Changes 0 to 10 for those who don't utilize prenatal care at all

clear data %drop original data matrix to save memory

%Dependent variable
y=bweight;

%Treatment variable
d=smoke;

%Covariates
x=[mage meduc prenatal prenatal_visits male married drink diabetes hyperpr amnio ultra terms>0];

% Save data for use with Python
save('y.mat','y')
save('d.mat','d')
save('X.mat','x')

%% OLS

X=[d, ones(length(d),1), x];
% beta
XTX=X'*X
XTX_inv=inv(XTX);
b1=XTX_inv*X';
betahat=b1*y
% residuals
uhat=y-X*betahat;
s2hat=uhat'*uhat/(N-14);
% covarianve
covhat=s2hat*XTX_inv
smokeSEhat=sqrt(covhat(1,1))

