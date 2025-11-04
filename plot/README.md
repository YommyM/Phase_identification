```text
./
-- input
|   |-- HMM                    # The phase state of each lipid identified by the HMM method
|   |   |-- dpdo280k
|   |   |-- dpdo290k
|   |   |-- dpdochl280k
|   |   |-- dpdochl290k
|   |   `-- psmdopochl
|   |-- full_time_scd          # The order parameter of each lipid over the entire trajectory
|   |-- last1us                # The normed density of each pixel
|   `-- last1us_gap5_scd_area  # The order parameters and APL of each lipid at the last 1 μs with 5 ns intervals
|-- output
|   |-- Fig2:densmap
|   |-- Fig3_S3-A:voronoi 
|   |-- Fig3_S3-BC:scd_area_joint_distri
|   |-- Fig4_S4:full_time
|   |-- FigS2:bin_choice
|   `-- threshold_tuning       # Visual Fine-Tuning
`-- scripts                    # All plotting scripts
```