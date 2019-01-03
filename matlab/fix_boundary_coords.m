function new_coords = fix_boundary_coords(dm_p, box_size , Cent)

num_particles = size(dm_p,1);

new_coords = zeros(size(dm_p));

for i=1:3
    CM_dist_0 = abs(dm_p(:,i) - Cent(i));
    CM_dist_1 = abs(dm_p(:,i) - ones(num_particles,1).*box_size - Cent(i));
    CM_dist_2 = abs(dm_p(:,i) + ones(num_particles,1).*box_size - Cent(i));
    [~ , J] = min([CM_dist_0 CM_dist_1 CM_dist_2],[],2);
    lin_ind = sub2ind([num_particles 3],(1:num_particles).',J);
    x_0 = dm_p(:,i);
    x_1 = dm_p(:,i) - ones(num_particles,1).*box_size;
    x_2 = dm_p(:,i) + ones(num_particles,1).*box_size;
    all_coords = [x_0 x_1 x_2];
    new_coords(:,i) = all_coords(lin_ind);    
end
