# mesh_rgbd_rendering
This repository takes a texture mesh (.obj, .ply) as its input, and outputs RGB-D images and pointclouds sampled from the texture mesh

1. Script1: generate view points, save intrinsics and cam2world poses in config.json file in folder data_path
    
    `./view_generator  data_path focal_length c_x c_y vertical_views horizontal_views`
2. Script2: visualize camera views saved in config.json file, data_path contains config.json file
    
    `./view_visualizer  data_path`
3. Script3: generate RGB-D images and point clouds saved in folder data_path
    
    `./mesh_rgbd_rendering_main data_path object_name`

    Depth images are saved unsigned short type (in mm)
