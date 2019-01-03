cd /mnt/home/yhezaveh/Illustris
basePath = '/mnt/ceph/users/sgenel/PUBLIC/Illustris-3/';
addpath(genpath('/mnt/home/yhezaveh/Illustris'));

subhalos = illustris.groupcat.loadSubhalos(basePath,135,fields);

header_data = illustris.snapshot.loadHeader( basePath , 135 );

fof_id = 1;
Cat_Stuff = illustris.groupcat.loadHalos(basePath,135,{'GroupFirstSub', 'GroupMass','GroupCM'});



box_size = header_data.BoxSize;

dm_p = illustris.snapshot.loadHalo( basePath , 135 , Cat_Stuff.GroupFirstSub(fof_id) , 'dm' , 'Coordinates' );
dm_p = dm_p.';

gas_p = illustris.snapshot.loadHalo( basePath , 135 , Cat_Stuff.GroupFirstSub(fof_id) , 'gas' , {'Coordinates','Density'} );
stars_p = illustris.snapshot.loadHalo( basePath , 135 , Cat_Stuff.GroupFirstSub(fof_id) , 'stars' , {'Coordinates','Masses'} );
BH_p = illustris.snapshot.loadHalo( basePath , 135 , Cat_Stuff.GroupFirstSub(fof_id) , 'stars' , {'Coordinates','Masses'} );

Cent = (Cat_Stuff.GroupCM).';


num_particles = size(dm_p,1);


new_coords = fix_boundary_coords(dm_p, box_size , Cent(fof_id,:));
[~ , D] = knnsearch(new_coords ,new_coords ,'K',10);


x = new_coords(:,1);
y = new_coords(:,2);
z = new_coords(:,3);
x_min = min(x);
x_max = max(x);
y_min = min(y);
y_max = max(y);
z_min = min(z);
z_max = max(z);
[X,Y] = meshgrid(linspace(x_min,x_max,1000),linspace(y_min,y_max,1000));
[X,Y,Z] = meshgrid(linspace(x_min,x_max,500),linspace(y_min,y_max,500),linspace(z_min,z_max,500));

I = 0;
for i=1:numel(x)
    i ./ numel(x)
    sigma = (mean(D(i,:),2));
    I = I + exp(-0.5 .* ((X-x(i)).^2+(Y-y(i)).^2+(Z-z(i)).^2)./sigma.^2)./sigma^3 ;
end

