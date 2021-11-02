[33mcommit 1e7a7c15ecaaac9c7cd755e18a9670d5002e64b9[m
Merge: 21aa828 6206510
Author: Alexandre Adam <adam.alexandre01213@gmail.com>
Date:   Tue Oct 26 08:42:42 2021 -0400

    Merge branch 'dev' of github.com:AlexandreAdam/Censai into dev

[33mcommit 21aa8289e214df4ccae5e098aa0ffb34dd45765a[m
Author: Alexandre Adam <adam.alexandre01213@gmail.com>
Date:   Tue Oct 26 08:42:34 2021 -0400

    Reverted change back to previous iteration

 censai/data/augmented_tng_kappa_generator.py        |  2 [32m+[m[31m-[m
 censai/data/delaunay_tessalation_field_estimator.py | 16 [32m+++++++++++[m[31m-----[m
 tests/test_physical_model.py                        | 93 [32m+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++[m[31m----------------------------------[m
 3 files changed, 71 insertions(+), 40 deletions(-)

[33mcommit 6206510537f9501f2042e1b9e26e2bd645cc5e03[m
Author: AlexandreAdam <adam.alexandre01123@gmail.com>
Date:   Tue Oct 26 00:11:03 2021 -0400

    Corrected changes in Rasterize halo

 scripts/rasterize_halo_kappa_maps.py | 6 [32m+++[m[31m---[m
 1 file changed, 3 insertions(+), 3 deletions(-)

[33mcommit 38ec014b3fc9a69e1057fb8cdf237833f33df84d[m
Merge: bedc681 87a32d4
Author: AlexandreAdam <adam.alexandre01123@gmail.com>
Date:   Tue Oct 26 00:08:08 2021 -0400

    merged

[33mcommit bedc681661efbcc8f23fdc8b18a38c477468e4ac[m
Author: AlexandreAdam <adam.alexandre01123@gmail.com>
Date:   Tue Oct 26 00:07:38 2021 -0400

    notebooks

 notebooks/cosmos_datasets.ipynb   | 6 [32m+++[m[31m---[m
 notebooks/rim_results_paper.ipynb | 4 [32m++[m[31m--[m
 requirements.txt                  | 2 [32m++[m
 3 files changed, 7 insertions(+), 5 deletions(-)

[33mcommit 87a32d498a111a5bdb26d702d488a6f6a1f49b61[m
Author: Alexandre Adam <adam.alexandre01213@gmail.com>
Date:   Mon Oct 25 21:12:03 2021 -0400

    Added variance and alpha variance to kappa fits files

 censai/data/delaunay_tessalation_field_estimator.py |  16 [32m++++++++[m
 notebooks/ivado_fig1.png                            | Bin [31m0[m -> [32m517919[m bytes
 notebooks/ivado_fig2.png                            | Bin [31m0[m -> [32m1314264[m bytes
 notebooks/ivado_fig3.png                            | Bin [31m0[m -> [32m1081033[m bytes
 notebooks/ivado_fig4.png                            | Bin [31m0[m -> [32m135285[m bytes
 notebooks/rim_finalrun128hst_results.ipynb          |  18 [32m++++[m[31m-----[m
 notebooks/rim_results_paper.ipynb                   | 336 [32m+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++[m[31m------------------------[m
 scripts/rasterize_halo_kappa_maps.py                |  27 [32m+++++++++[m[31m----[m
 8 files changed, 329 insertions(+), 68 deletions(-)

[33mcommit 2f90931b63b6a632d36c5ebf7dc355d2dd99f706[m
Author: Alexandre Adam <adam.alexandre01213@gmail.com>
Date:   Mon Oct 25 10:12:04 2021 -0400

    Reverted back change to loglikelihood in v0 version of rim shared unet

 censai/rim_shared_unet.py          | 12 [32m++++++[m[31m------[m
 scripts/train_rim_shared_unet.py   |  4 [32m++[m[31m--[m
 scripts/train_rim_shared_unetv2.py |  4 [32m++[m[31m--[m
 scripts/train_rim_shared_unetv3.py |  4 [32m++[m[31m--[m
 scripts/train_rim_unet.py          |  4 [32m++[m[31m--[m
 5 files changed, 14 insertions(+), 14 deletions(-)

[33mcommit ced2ffae0b33bc76116bcc936bf24d501c784ef2[m
Author: Alexandre Adam <adam.alexandre01213@gmail.com>
Date:   Mon Oct 25 01:05:05 2021 -0400

    Changed index of nearest neighbor to maximum distance since neighbors aren't sorted

 scripts/rasterize_halo_kappa_maps.py | 2 [32m+[m[31m-[m
 1 file changed, 1 insertion(+), 1 deletion(-)

[33mcommit c45739d27a9392ab588d1f115c8e0b64fe87d8f8[m
Author: Alexandre Adam <adam.alexandre01213@gmail.com>
Date:   Mon Oct 25 00:05:41 2021 -0400

    .

 censai/rim_shared_unet.py                                       | 12 [32m++++++[m[31m------[m
 scripts/experiments/rim_unet_gridsearch.py                      |  1 [32m+[m
 scripts/rasterize_halo_kappa_maps.py                            | 18 [32m++[m[31m----------------[m
 scripts/shell/final_run128HST/1_rasterize_halo_kappa_maps_xy.sh |  2 [32m+[m[31m-[m
 scripts/shell/final_run128HST/1_rasterize_halo_kappa_maps_xz.sh |  2 [32m+[m[31m-[m
 scripts/shell/final_run128HST/1_rasterize_halo_kappa_maps_yz.sh |  3 [32m+[m[31m--[m
 scripts/train_rim_unet.py                                       | 73 [32m++++++++++++++++++++++++++++++++++++++++++++++++++++++++++[m[31m---------------[m
 7 files changed, 70 insertions(+), 41 deletions(-)

[33mcommit 845d3df00f1e0b22ae83977f09162855de2de5c0[m
Author: Alexandre Adam <adam.alexandre01213@gmail.com>
Date:   Sun Oct 24 23:18:31 2021 -0400

    Added v3 script with log_likelihood masked

 censai/__init__.py                                                 |   1 [32m+[m
 censai/physical_model.py                                           |   7 [32m++[m
 censai/rim_shared_unetv2.py                                        | 228 [32m+++++++++++++++++++++++++++++++++++++++++++++++++++++[m
 scripts/shell/final_run128HST/5_4_rim_shared_unetv3_control_ts8.sh |  59 [32m++++++++++++++[m
 scripts/train_rim_shared_unetv3.py                                 | 588 [32m++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++[m
 5 files changed, 883 insertions(+)

[33mcommit 59c4e118fd9fa20e3fb6cb2d322ef24431a68768[m
Merge: d73865a 306bd09
Author: Alexandre Adam <aadam@beluga4.int.ets1.calculquebec.ca>
Date:   Sun Oct 24 23:05:58 2021 -0400

    Merge branch 'dev' of github.com:AlexandreAdam/Censai into dev

[33mcommit d73865a180059f5edcd0d9e74572965af4a703d1[m
Merge: 38100af c670ae6
Author: Alexandre Adam <aadam@beluga4.int.ets1.calculquebec.ca>
Date:   Sun Oct 24 23:04:17 2021 -0400

    new scripts beluga

[33mcommit 306bd093bd70c50dc1ef234b3c0b67a4daa8166d[m
Author: Alexandre Adam <adam.alexandre01213@gmail.com>
Date:   Sun Oct 24 22:03:26 2021 -0400

    .

 censai/models/layers/conv_encoding_layer.py                                         |  7 [32m++++++[m[31m-[m
 scripts/shell/scale_complexity_128/3_halo/create_halos_noshift_verydiffuse_noisy.sh | 37 [32m+++++++++++++++++++++++++++++++++++++[m
 2 files changed, 43 insertions(+), 1 deletion(-)
