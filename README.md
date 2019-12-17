# object_rendering

1. Script1: generate view points, save intrinsics and cam2world poses in config.json file in folder data_path
    
    `./view_generator  data_path focal_length c_x c_y vertical_views horizontal_views`
2. Script2: visualize camera views saved in config.json file, data_path contains config.json file
    
    `./object_visualizer  data_path`
3. Script3: generate RGB-D images and point clouds saved in folder data_path
    
    `./object_rendering data_path object_name`

    Depth images are saved unsigned short type (in mm)
