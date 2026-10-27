# Cross-check report

## vs QuantLib golden
| case | lib | lib price | golden | rel err | tol | verdict |
|---|---|---|---|---|---|---|
| bsm_eu_call | quantburn-cpu | 9.227005508 | 9.227005508 | 2.70e-15 | 1e-06 | PASS |
| bsm_eu_put | quantburn-cpu | 6.330080628 | 6.330080628 | 1.40e-16 | 1e-06 | PASS |
| crr_eu_call | quantburn-cpu | 10.43061166 | 10.42998595 | 6.00e-05 | 1e-04 | PASS |
| crr_amer_put | quantburn-cpu | 6.082354409 | 6.082618218 | 4.34e-05 | 1e-04 | PASS |
| amer_put_100 | quantburn-cpu | 6.082354409 | 6.082618218 | 4.34e-05 | 1e-04 | PASS |
| heston_cf_call | quantburn-cpu | 9.897022477 | 9.897035612 | 1.33e-06 | 1e-05 | PASS |
| sabr_iv | quantburn-cpu | 0.0736879502 | 0.0736879502 | 0.00e+00 | 1e-06 | PASS |
| black76_call | quantburn-cpu | 7.577082146 | 7.577082146 | 0.00e+00 | 1e-06 | PASS |
| vasicek_zcb_1y | quantburn-cpu | 0.9691188083 | 0.9691188083 | 0.00e+00 | 1e-06 | PASS |
| vasicek_zcb_5y | quantburn-cpu | 0.8396917673 | 0.8396917673 | 0.00e+00 | 1e-06 | PASS |
| cir_zcb_1y | quantburn-cpu | 0.9690627652 | 0.9690627652 | 0.00e+00 | 1e-06 | PASS |
| hw_zcb_1y | quantburn-cpu | 0.9704455335 | 0.9704455335 | 8.38e-13 | 1e-06 | PASS |
| tfq_iv_newton_call | quantburn-cpu | (unknown model 'iv') | 0.2 | — | — | SKIP |
| tfq_iv_newton_put | quantburn-cpu | (unknown model 'iv') | 0.35 | — | — | SKIP |
| tfq_iv_approx_atm | quantburn-cpu | (unknown model 'iv') | 0.2 | — | — | SKIP |
| tfq_asian_geo_m | quantburn-cpu | (unknown model 'asian_analytic') | 5.289698532 | — | — | SKIP |
| tfq_barrier_doko_call | quantburn-cpu | (unknown model 'barrier') | 8.759460459 | — | — | SKIP |
| tfq_barrier_uoko_put | quantburn-cpu | (unknown model 'barrier') | 7.902600854 | — | — | SKIP |
| tfq_barrier_diki_call | quantburn-cpu | (unknown model 'barrier') | 0.5479573226 | — | — | SKIP |
| tfq_barrier_uiki_put | quantburn-cpu | (unknown model 'barrier') | 0.01342299254 | — | — | SKIP |
| tfq_binary_call | quantburn-cpu | (unknown model 'binary') | 0.4945810911 | — | — | SKIP |
| tfq_binary_put | quantburn-cpu | (unknown model 'binary') | 0.5575066596 | — | — | SKIP |
| tfq_pde_eu_call | quantburn-cpu | (unknown model 'pde_eu') | 9.227498466 | — | — | SKIP |
| tfq_swapcurve_flat | quantburn-cpu | (unknown model 'swap_curve') | 0.9924722288 | — | — | SKIP |
| bsm_eu_call | quantburn-vulkan | 9.227005508 | 9.227005508 | 2.70e-15 | 1e-06 | PASS |
| bsm_eu_put | quantburn-vulkan | 6.330080628 | 6.330080628 | 1.40e-16 | 1e-06 | PASS |
| crr_eu_call | quantburn-vulkan | 10.43061166 | 10.42998595 | 6.00e-05 | 1e-04 | PASS |
| crr_amer_put | quantburn-vulkan | 6.082354409 | 6.082618218 | 4.34e-05 | 1e-04 | PASS |
| amer_put_100 | quantburn-vulkan | 6.082354409 | 6.082618218 | 4.34e-05 | 1e-04 | PASS |
| heston_cf_call | quantburn-vulkan | 9.897022477 | 9.897035612 | 1.33e-06 | 1e-05 | PASS |
| sabr_iv | quantburn-vulkan | 0.0736879502 | 0.0736879502 | 0.00e+00 | 1e-06 | PASS |
| black76_call | quantburn-vulkan | 7.577082146 | 7.577082146 | 0.00e+00 | 1e-06 | PASS |
| vasicek_zcb_1y | quantburn-vulkan | 0.9691188083 | 0.9691188083 | 0.00e+00 | 1e-06 | PASS |
| vasicek_zcb_5y | quantburn-vulkan | 0.8396917673 | 0.8396917673 | 0.00e+00 | 1e-06 | PASS |
| cir_zcb_1y | quantburn-vulkan | 0.9690627652 | 0.9690627652 | 0.00e+00 | 1e-06 | PASS |
| hw_zcb_1y | quantburn-vulkan | 0.9704455335 | 0.9704455335 | 8.38e-13 | 1e-06 | PASS |
| tfq_iv_newton_call | quantburn-vulkan | (unknown model 'iv') | 0.2 | — | — | SKIP |
| tfq_iv_newton_put | quantburn-vulkan | (unknown model 'iv') | 0.35 | — | — | SKIP |
| tfq_iv_approx_atm | quantburn-vulkan | (unknown model 'iv') | 0.2 | — | — | SKIP |
| tfq_asian_geo_m | quantburn-vulkan | (unknown model 'asian_analytic') | 5.289698532 | — | — | SKIP |
| tfq_barrier_doko_call | quantburn-vulkan | (unknown model 'barrier') | 8.759460459 | — | — | SKIP |
| tfq_barrier_uoko_put | quantburn-vulkan | (unknown model 'barrier') | 7.902600854 | — | — | SKIP |
| tfq_barrier_diki_call | quantburn-vulkan | (unknown model 'barrier') | 0.5479573226 | — | — | SKIP |
| tfq_barrier_uiki_put | quantburn-vulkan | (unknown model 'barrier') | 0.01342299254 | — | — | SKIP |
| tfq_binary_call | quantburn-vulkan | (unknown model 'binary') | 0.4945810911 | — | — | SKIP |
| tfq_binary_put | quantburn-vulkan | (unknown model 'binary') | 0.5575066596 | — | — | SKIP |
| tfq_pde_eu_call | quantburn-vulkan | (unknown model 'pde_eu') | 9.227498466 | — | — | SKIP |
| tfq_swapcurve_flat | quantburn-vulkan | (unknown model 'swap_curve') | 0.9924722288 | — | — | SKIP |
| bsm_eu_call | quantcandle-cpu | 9.227005508 | 9.227005508 | 2.70e-15 | 1e-06 | PASS |
| bsm_eu_put | quantcandle-cpu | 6.330080628 | 6.330080628 | 1.40e-16 | 1e-06 | PASS |
| crr_eu_call | quantcandle-cpu | 10.43061166 | 10.42998595 | 6.00e-05 | 1e-04 | PASS |
| crr_amer_put | quantcandle-cpu | 6.082354409 | 6.082618218 | 4.34e-05 | 1e-04 | PASS |
| amer_put_100 | quantcandle-cpu | 6.082354409 | 6.082618218 | 4.34e-05 | 1e-04 | PASS |
| heston_cf_call | quantcandle-cpu | 9.897022477 | 9.897035612 | 1.33e-06 | 1e-05 | PASS |
| sabr_iv | quantcandle-cpu | 0.0736879502 | 0.0736879502 | 0.00e+00 | 1e-06 | PASS |
| black76_call | quantcandle-cpu | 7.577082146 | 7.577082146 | 0.00e+00 | 1e-06 | PASS |
| vasicek_zcb_1y | quantcandle-cpu | 0.9691188083 | 0.9691188083 | 0.00e+00 | 1e-06 | PASS |
| vasicek_zcb_5y | quantcandle-cpu | 0.8396917673 | 0.8396917673 | 0.00e+00 | 1e-06 | PASS |
| cir_zcb_1y | quantcandle-cpu | 0.9690627652 | 0.9690627652 | 0.00e+00 | 1e-06 | PASS |
| hw_zcb_1y | quantcandle-cpu | 0.9704455335 | 0.9704455335 | 8.38e-13 | 1e-06 | PASS |
| tfq_iv_newton_call | quantcandle-cpu | (unknown model 'iv') | 0.2 | — | — | SKIP |
| tfq_iv_newton_put | quantcandle-cpu | (unknown model 'iv') | 0.35 | — | — | SKIP |
| tfq_iv_approx_atm | quantcandle-cpu | (unknown model 'iv') | 0.2 | — | — | SKIP |
| tfq_asian_geo_m | quantcandle-cpu | (unknown model 'asian_analytic') | 5.289698532 | — | — | SKIP |
| tfq_barrier_doko_call | quantcandle-cpu | (unknown model 'barrier') | 8.759460459 | — | — | SKIP |
| tfq_barrier_uoko_put | quantcandle-cpu | (unknown model 'barrier') | 7.902600854 | — | — | SKIP |
| tfq_barrier_diki_call | quantcandle-cpu | (unknown model 'barrier') | 0.5479573226 | — | — | SKIP |
| tfq_barrier_uiki_put | quantcandle-cpu | (unknown model 'barrier') | 0.01342299254 | — | — | SKIP |
| tfq_binary_call | quantcandle-cpu | (unknown model 'binary') | 0.4945810911 | — | — | SKIP |
| tfq_binary_put | quantcandle-cpu | (unknown model 'binary') | 0.5575066596 | — | — | SKIP |
| tfq_pde_eu_call | quantcandle-cpu | (unknown model 'pde_eu') | 9.227498466 | — | — | SKIP |
| tfq_swapcurve_flat | quantcandle-cpu | (unknown model 'swap_curve') | 0.9924722288 | — | — | SKIP |
| bsm_eu_call | quantcandle-mkl | 9.227005508 | 9.227005508 | 2.70e-15 | 1e-06 | PASS |
| bsm_eu_put | quantcandle-mkl | 6.330080628 | 6.330080628 | 1.40e-16 | 1e-06 | PASS |
| crr_eu_call | quantcandle-mkl | 10.43061166 | 10.42998595 | 6.00e-05 | 1e-04 | PASS |
| crr_amer_put | quantcandle-mkl | 6.082354409 | 6.082618218 | 4.34e-05 | 1e-04 | PASS |
| amer_put_100 | quantcandle-mkl | 6.082354409 | 6.082618218 | 4.34e-05 | 1e-04 | PASS |
| heston_cf_call | quantcandle-mkl | 9.897022477 | 9.897035612 | 1.33e-06 | 1e-05 | PASS |
| sabr_iv | quantcandle-mkl | 0.0736879502 | 0.0736879502 | 0.00e+00 | 1e-06 | PASS |
| black76_call | quantcandle-mkl | 7.577082146 | 7.577082146 | 0.00e+00 | 1e-06 | PASS |
| vasicek_zcb_1y | quantcandle-mkl | 0.9691188083 | 0.9691188083 | 0.00e+00 | 1e-06 | PASS |
| vasicek_zcb_5y | quantcandle-mkl | 0.8396917673 | 0.8396917673 | 0.00e+00 | 1e-06 | PASS |
| cir_zcb_1y | quantcandle-mkl | 0.9690627652 | 0.9690627652 | 0.00e+00 | 1e-06 | PASS |
| hw_zcb_1y | quantcandle-mkl | 0.9704455335 | 0.9704455335 | 8.38e-13 | 1e-06 | PASS |
| tfq_iv_newton_call | quantcandle-mkl | (unknown model 'iv') | 0.2 | — | — | SKIP |
| tfq_iv_newton_put | quantcandle-mkl | (unknown model 'iv') | 0.35 | — | — | SKIP |
| tfq_iv_approx_atm | quantcandle-mkl | (unknown model 'iv') | 0.2 | — | — | SKIP |
| tfq_asian_geo_m | quantcandle-mkl | (unknown model 'asian_analytic') | 5.289698532 | — | — | SKIP |
| tfq_barrier_doko_call | quantcandle-mkl | (unknown model 'barrier') | 8.759460459 | — | — | SKIP |
| tfq_barrier_uoko_put | quantcandle-mkl | (unknown model 'barrier') | 7.902600854 | — | — | SKIP |
| tfq_barrier_diki_call | quantcandle-mkl | (unknown model 'barrier') | 0.5479573226 | — | — | SKIP |
| tfq_barrier_uiki_put | quantcandle-mkl | (unknown model 'barrier') | 0.01342299254 | — | — | SKIP |
| tfq_binary_call | quantcandle-mkl | (unknown model 'binary') | 0.4945810911 | — | — | SKIP |
| tfq_binary_put | quantcandle-mkl | (unknown model 'binary') | 0.5575066596 | — | — | SKIP |
| tfq_pde_eu_call | quantcandle-mkl | (unknown model 'pde_eu') | 9.227498466 | — | — | SKIP |
| tfq_swapcurve_flat | quantcandle-mkl | (unknown model 'swap_curve') | 0.9924722288 | — | — | SKIP |
| bsm_eu_call | quantorch-cpu | 9.227005508 | 9.227005508 | 2.70e-15 | 1e-05 | PASS |
| bsm_eu_put | quantorch-cpu | 6.330080628 | 6.330080628 | 1.40e-16 | 1e-05 | PASS |
| crr_eu_call | quantorch-cpu | 10.43061166 | 10.42998595 | 6.00e-05 | 1e-04 | PASS |
| crr_amer_put | quantorch-cpu | 6.082354409 | 6.082618218 | 4.34e-05 | 1e-04 | PASS |
| amer_put_100 | quantorch-cpu | 6.082354409 | 6.082618218 | 4.34e-05 | 1e-04 | PASS |
| heston_cf_call | quantorch-cpu | 9.897022247 | 9.897035612 | 1.35e-06 | 1e-05 | PASS |
| sabr_iv | quantorch-cpu | 0.07368795574 | 0.0736879502 | 7.52e-08 | 1e-05 | PASS |
| black76_call | quantorch-cpu | 7.577082146 | 7.577082146 | 0.00e+00 | 1e-05 | PASS |
| vasicek_zcb_1y | quantorch-cpu | 0.9691188335 | 0.9691188083 | 2.61e-08 | 1e-05 | PASS |
| vasicek_zcb_5y | quantorch-cpu | 0.8396917582 | 0.8396917673 | 1.09e-08 | 1e-05 | PASS |
| cir_zcb_1y | quantorch-cpu | 0.9690644145 | 0.9690627652 | 1.70e-06 | 1e-05 | PASS |
| hw_zcb_1y | quantorch-cpu | 0.9704455137 | 0.9704455335 | 2.04e-08 | 1e-05 | PASS |
| tfq_iv_newton_call | quantorch-cpu | (ValueError: unknown model 'iv') | 0.2 | — | — | SKIP |
| tfq_iv_newton_put | quantorch-cpu | (ValueError: unknown model 'iv') | 0.35 | — | — | SKIP |
| tfq_iv_approx_atm | quantorch-cpu | (ValueError: unknown model 'iv') | 0.2 | — | — | SKIP |
| tfq_asian_geo_m | quantorch-cpu | (ValueError: unknown model 'asian_analytic') | 5.289698532 | — | — | SKIP |
| tfq_barrier_doko_call | quantorch-cpu | (ValueError: unknown model 'barrier') | 8.759460459 | — | — | SKIP |
| tfq_barrier_uoko_put | quantorch-cpu | (ValueError: unknown model 'barrier') | 7.902600854 | — | — | SKIP |
| tfq_barrier_diki_call | quantorch-cpu | (ValueError: unknown model 'barrier') | 0.5479573226 | — | — | SKIP |
| tfq_barrier_uiki_put | quantorch-cpu | (ValueError: unknown model 'barrier') | 0.01342299254 | — | — | SKIP |
| tfq_binary_call | quantorch-cpu | (ValueError: unknown model 'binary') | 0.4945810911 | — | — | SKIP |
| tfq_binary_put | quantorch-cpu | (ValueError: unknown model 'binary') | 0.5575066596 | — | — | SKIP |
| tfq_pde_eu_call | quantorch-cpu | (ValueError: unknown model 'pde_eu') | 9.227498466 | — | — | SKIP |
| tfq_swapcurve_flat | quantorch-cpu | (ValueError: unknown model 'swap_curve') | 0.9924722288 | — | — | SKIP |
| bsm_eu_call | tfquant-gpu | 9.227005508 | 9.227005508 | 0.00e+00 | 1e-06 | PASS |
| bsm_eu_put | tfquant-gpu | 6.330080628 | 6.330080628 | 1.40e-16 | 1e-06 | PASS |
| crr_eu_call | tfquant-gpu | 10.43061166 | 10.42998595 | 6.00e-05 | 1e-04 | PASS |
| crr_amer_put | tfquant-gpu | 6.082354409 | 6.082618218 | 4.34e-05 | 1e-04 | PASS |
| amer_put_100 | tfquant-gpu | 6.082354409 | 6.082618218 | 4.34e-05 | 1e-04 | PASS |
| heston_cf_call | tfquant-gpu | (no tf-quant equivalent in scope) | 9.897035612 | — | — | SKIP |
| sabr_iv | tfquant-gpu | (no tf-quant equivalent in scope) | 0.0736879502 | — | — | SKIP |
| black76_call | tfquant-gpu | (no tf-quant equivalent in scope) | 7.577082146 | — | — | SKIP |
| vasicek_zcb_1y | tfquant-gpu | (no tf-quant equivalent in scope) | 0.9691188083 | — | — | SKIP |
| vasicek_zcb_5y | tfquant-gpu | (no tf-quant equivalent in scope) | 0.8396917673 | — | — | SKIP |
| cir_zcb_1y | tfquant-gpu | (no tf-quant equivalent in scope) | 0.9690627652 | — | — | SKIP |
| hw_zcb_1y | tfquant-gpu | (no tf-quant equivalent in scope) | 0.9704455335 | — | — | SKIP |
| tfq_iv_newton_call | tfquant-gpu | 0.1999999993 | 0.2 | 3.29e-09 | 1e-06 | PASS |
| tfq_iv_newton_put | tfquant-gpu | 0.3499999919 | 0.35 | 2.31e-08 | 1e-06 | PASS |
| tfq_iv_approx_atm | tfquant-gpu | 0.1998836711 | 0.2 | 5.82e-04 | 1e-03 | PASS |
| tfq_asian_geo_m | tfquant-gpu | 5.289698532 | 5.289698532 | 7.20e-12 | 1e-06 | PASS |
| tfq_barrier_doko_call | tfquant-gpu | 8.759460459 | 8.759460459 | 8.11e-16 | 1e-06 | PASS |
| tfq_barrier_uoko_put | tfquant-gpu | 7.902600854 | 7.902600854 | 1.12e-15 | 1e-06 | PASS |
| tfq_barrier_diki_call | tfquant-gpu | 0.5479573226 | 0.5479573226 | 9.73e-15 | 1e-06 | PASS |
| tfq_barrier_uiki_put | tfquant-gpu | 0.01342299254 | 0.01342299254 | 6.20e-15 | 1e-06 | PASS |
| tfq_binary_call | tfquant-gpu | 0.4945810911 | 0.4945810911 | 7.86e-16 | 1e-06 | PASS |
| tfq_binary_put | tfquant-gpu | 0.5575066596 | 0.5575066596 | 0.00e+00 | 1e-06 | PASS |
| tfq_pde_eu_call | tfquant-gpu | 9.22565548 | 9.227498466 | 2.00e-04 | 5e-03 | PASS |
| tfq_swapcurve_flat | tfquant-gpu | 0.9924722288 | 0.9924722288 | 3.12e-07 | 1e-06 | PASS |

## library vs library (same case ids)
| case | a | b | a price | b price | rel err | verdict |
|---|---|---|---|---|---|---|
| bsm_eu_call | quantburn-cpu | quantburn-vulkan | 9.227005508 | 9.227005508 | 0.00e+00 | PASS |
| bsm_eu_put | quantburn-cpu | quantburn-vulkan | 6.330080628 | 6.330080628 | 0.00e+00 | PASS |
| crr_eu_call | quantburn-cpu | quantburn-vulkan | 10.43061166 | 10.43061166 | 0.00e+00 | PASS |
| crr_amer_put | quantburn-cpu | quantburn-vulkan | 6.082354409 | 6.082354409 | 0.00e+00 | PASS |
| amer_put_100 | quantburn-cpu | quantburn-vulkan | 6.082354409 | 6.082354409 | 0.00e+00 | PASS |
| berm_put | quantburn-cpu | quantburn-vulkan | 6.025795352 | 6.025795352 | 0.00e+00 | PASS |
| asian_call | quantburn-cpu | quantburn-vulkan | 6.62561789 | 6.62561789 | 0.00e+00 | PASS(rng) |
| heston_cf_call | quantburn-cpu | quantburn-vulkan | 9.897022477 | 9.897022477 | 0.00e+00 | PASS |
| heston_mc_call | quantburn-cpu | quantburn-vulkan | 9.586220714 | 9.586220714 | 0.00e+00 | PASS(rng) |
| sabr_iv | quantburn-cpu | quantburn-vulkan | 0.0736879502 | 0.0736879502 | 0.00e+00 | PASS |
| sabr_price | quantburn-cpu | quantburn-vulkan | 0.9367717118 | 0.9367717118 | 0.00e+00 | PASS |
| black76_call | quantburn-cpu | quantburn-vulkan | 7.577082146 | 7.577082146 | 0.00e+00 | PASS |
| vasicek_zcb_1y | quantburn-cpu | quantburn-vulkan | 0.9691188083 | 0.9691188083 | 0.00e+00 | PASS |
| vasicek_zcb_5y | quantburn-cpu | quantburn-vulkan | 0.8396917673 | 0.8396917673 | 0.00e+00 | PASS |
| cir_zcb_1y | quantburn-cpu | quantburn-vulkan | 0.9690627652 | 0.9690627652 | 0.00e+00 | PASS |
| hw_zcb_1y | quantburn-cpu | quantburn-vulkan | 0.9704455335 | 0.9704455335 | 0.00e+00 | PASS |
| bk_zcb_5y | quantburn-cpu | quantburn-vulkan | 0.7105591542 | 0.7105591542 | 0.00e+00 | PASS |
| digital_call | quantburn-cpu | quantburn-vulkan | 5.796554306 | 5.796554306 | 0.00e+00 | PASS(rng) |
| barrier_uo | quantburn-cpu | quantburn-vulkan | 1.730618702 | 1.730618702 | 0.00e+00 | PASS(rng) |
| lookback_call | quantburn-cpu | quantburn-vulkan | 17.65824239 | 17.65824239 | 0.00e+00 | PASS(rng) |
| chooser | quantburn-cpu | quantburn-vulkan | 6.718645263 | 6.718645263 | 0.00e+00 | PASS |
| bsm_eu_call | quantburn-cpu | quantcandle-cpu | 9.227005508 | 9.227005508 | 0.00e+00 | PASS |
| bsm_eu_put | quantburn-cpu | quantcandle-cpu | 6.330080628 | 6.330080628 | 0.00e+00 | PASS |
| crr_eu_call | quantburn-cpu | quantcandle-cpu | 10.43061166 | 10.43061166 | 0.00e+00 | PASS |
| crr_amer_put | quantburn-cpu | quantcandle-cpu | 6.082354409 | 6.082354409 | 0.00e+00 | PASS |
| amer_put_100 | quantburn-cpu | quantcandle-cpu | 6.082354409 | 6.082354409 | 0.00e+00 | PASS |
| berm_put | quantburn-cpu | quantcandle-cpu | 6.025795352 | 6.025795352 | 0.00e+00 | PASS |
| asian_call | quantburn-cpu | quantcandle-cpu | 6.62561789 | 6.62561789 | 0.00e+00 | PASS(rng) |
| heston_cf_call | quantburn-cpu | quantcandle-cpu | 9.897022477 | 9.897022477 | 0.00e+00 | PASS |
| heston_mc_call | quantburn-cpu | quantcandle-cpu | 9.586220714 | 9.586220714 | 0.00e+00 | PASS(rng) |
| sabr_iv | quantburn-cpu | quantcandle-cpu | 0.0736879502 | 0.0736879502 | 0.00e+00 | PASS |
| sabr_price | quantburn-cpu | quantcandle-cpu | 0.9367717118 | 0.9367717118 | 0.00e+00 | PASS |
| black76_call | quantburn-cpu | quantcandle-cpu | 7.577082146 | 7.577082146 | 0.00e+00 | PASS |
| vasicek_zcb_1y | quantburn-cpu | quantcandle-cpu | 0.9691188083 | 0.9691188083 | 0.00e+00 | PASS |
| vasicek_zcb_5y | quantburn-cpu | quantcandle-cpu | 0.8396917673 | 0.8396917673 | 0.00e+00 | PASS |
| cir_zcb_1y | quantburn-cpu | quantcandle-cpu | 0.9690627652 | 0.9690627652 | 0.00e+00 | PASS |
| hw_zcb_1y | quantburn-cpu | quantcandle-cpu | 0.9704455335 | 0.9704455335 | 0.00e+00 | PASS |
| bk_zcb_5y | quantburn-cpu | quantcandle-cpu | 0.7105591542 | 0.7105591542 | 0.00e+00 | PASS |
| digital_call | quantburn-cpu | quantcandle-cpu | 5.796554306 | 5.796554306 | 0.00e+00 | PASS(rng) |
| barrier_uo | quantburn-cpu | quantcandle-cpu | 1.730618702 | 1.730618702 | 0.00e+00 | PASS(rng) |
| lookback_call | quantburn-cpu | quantcandle-cpu | 17.65824239 | 17.65824239 | 0.00e+00 | PASS(rng) |
| chooser | quantburn-cpu | quantcandle-cpu | 6.718645263 | 6.718645263 | 0.00e+00 | PASS |
| bsm_eu_call | quantburn-cpu | quantcandle-mkl | 9.227005508 | 9.227005508 | 0.00e+00 | PASS |
| bsm_eu_put | quantburn-cpu | quantcandle-mkl | 6.330080628 | 6.330080628 | 0.00e+00 | PASS |
| crr_eu_call | quantburn-cpu | quantcandle-mkl | 10.43061166 | 10.43061166 | 0.00e+00 | PASS |
| crr_amer_put | quantburn-cpu | quantcandle-mkl | 6.082354409 | 6.082354409 | 0.00e+00 | PASS |
| amer_put_100 | quantburn-cpu | quantcandle-mkl | 6.082354409 | 6.082354409 | 0.00e+00 | PASS |
| berm_put | quantburn-cpu | quantcandle-mkl | 6.025795352 | 6.025795352 | 0.00e+00 | PASS |
| asian_call | quantburn-cpu | quantcandle-mkl | 6.62561789 | 6.62561789 | 0.00e+00 | PASS(rng) |
| heston_cf_call | quantburn-cpu | quantcandle-mkl | 9.897022477 | 9.897022477 | 0.00e+00 | PASS |
| heston_mc_call | quantburn-cpu | quantcandle-mkl | 9.586220714 | 9.586220714 | 0.00e+00 | PASS(rng) |
| sabr_iv | quantburn-cpu | quantcandle-mkl | 0.0736879502 | 0.0736879502 | 0.00e+00 | PASS |
| sabr_price | quantburn-cpu | quantcandle-mkl | 0.9367717118 | 0.9367717118 | 0.00e+00 | PASS |
| black76_call | quantburn-cpu | quantcandle-mkl | 7.577082146 | 7.577082146 | 0.00e+00 | PASS |
| vasicek_zcb_1y | quantburn-cpu | quantcandle-mkl | 0.9691188083 | 0.9691188083 | 0.00e+00 | PASS |
| vasicek_zcb_5y | quantburn-cpu | quantcandle-mkl | 0.8396917673 | 0.8396917673 | 0.00e+00 | PASS |
| cir_zcb_1y | quantburn-cpu | quantcandle-mkl | 0.9690627652 | 0.9690627652 | 0.00e+00 | PASS |
| hw_zcb_1y | quantburn-cpu | quantcandle-mkl | 0.9704455335 | 0.9704455335 | 0.00e+00 | PASS |
| bk_zcb_5y | quantburn-cpu | quantcandle-mkl | 0.7105591542 | 0.7105591542 | 0.00e+00 | PASS |
| digital_call | quantburn-cpu | quantcandle-mkl | 5.796554306 | 5.796554306 | 0.00e+00 | PASS(rng) |
| barrier_uo | quantburn-cpu | quantcandle-mkl | 1.730618702 | 1.730618702 | 0.00e+00 | PASS(rng) |
| lookback_call | quantburn-cpu | quantcandle-mkl | 17.65824239 | 17.65824239 | 0.00e+00 | PASS(rng) |
| chooser | quantburn-cpu | quantcandle-mkl | 6.718645263 | 6.718645263 | 0.00e+00 | PASS |
| bsm_eu_call | quantburn-cpu | quantorch-cpu | 9.227005508 | 9.227005508 | 0.00e+00 | PASS |
| bsm_eu_put | quantburn-cpu | quantorch-cpu | 6.330080628 | 6.330080628 | 0.00e+00 | PASS |
| crr_eu_call | quantburn-cpu | quantorch-cpu | 10.43061166 | 10.43061166 | 3.07e-15 | PASS |
| crr_amer_put | quantburn-cpu | quantorch-cpu | 6.082354409 | 6.082354409 | 2.92e-15 | PASS |
| amer_put_100 | quantburn-cpu | quantorch-cpu | 6.082354409 | 6.082354409 | 2.92e-15 | PASS |
| berm_put | quantburn-cpu | quantorch-cpu | 6.025795352 | 6.025795352 | 2.36e-15 | PASS |
| asian_call | quantburn-cpu | quantorch-cpu | 6.62561789 | 7.020513535 | 5.62e-02 | INFO(rng) |
| heston_cf_call | quantburn-cpu | quantorch-cpu | 9.897022477 | 9.897022247 | 2.32e-08 | PASS(f32) |
| heston_mc_call | quantburn-cpu | quantorch-cpu | 9.586220714 | 9.927351952 | 3.44e-02 | INFO(rng) |
| sabr_iv | quantburn-cpu | quantorch-cpu | 0.0736879502 | 0.07368795574 | 7.52e-08 | PASS(f32) |
| sabr_price | quantburn-cpu | quantorch-cpu | 0.9367717118 | 0.9367747307 | 3.22e-06 | PASS(f32) |
| black76_call | quantburn-cpu | quantorch-cpu | 7.577082146 | 7.577082146 | 0.00e+00 | PASS |
| vasicek_zcb_1y | quantburn-cpu | quantorch-cpu | 0.9691188083 | 0.9691188335 | 2.61e-08 | PASS(f32) |
| vasicek_zcb_5y | quantburn-cpu | quantorch-cpu | 0.8396917673 | 0.8396917582 | 1.09e-08 | PASS(f32) |
| cir_zcb_1y | quantburn-cpu | quantorch-cpu | 0.9690627652 | 0.9690644145 | 1.70e-06 | PASS(f32) |
| hw_zcb_1y | quantburn-cpu | quantorch-cpu | 0.9704455335 | 0.9704455137 | 2.04e-08 | PASS(f32) |
| bk_zcb_5y | quantburn-cpu | quantorch-cpu | 0.7105591542 | 0.7105591893 | 4.95e-08 | PASS(f32) |
| digital_call | quantburn-cpu | quantorch-cpu | 5.796554306 | 5.164878845 | 1.22e-01 | INFO(rng) |
| barrier_uo | quantburn-cpu | quantorch-cpu | 1.730618702 | 1.089030385 | 5.89e-01 | INFO(rng) |
| lookback_call | quantburn-cpu | quantorch-cpu | 17.65824239 | 20.5508728 | 1.41e-01 | INFO(rng) |
| chooser | quantburn-cpu | quantorch-cpu | 6.718645263 | 6.718645263 | 0.00e+00 | PASS |
| bsm_eu_call | quantburn-cpu | tfquant-gpu | 9.227005508 | 9.227005508 | 2.70e-15 | PASS |
| bsm_eu_put | quantburn-cpu | tfquant-gpu | 6.330080628 | 6.330080628 | 0.00e+00 | PASS |
| crr_eu_call | quantburn-cpu | tfquant-gpu | 10.43061166 | 10.43061166 | 5.65e-14 | PASS |
| crr_amer_put | quantburn-cpu | tfquant-gpu | 6.082354409 | 6.082354409 | 1.39e-13 | PASS |
| amer_put_100 | quantburn-cpu | tfquant-gpu | 6.082354409 | 6.082354409 | 1.39e-13 | PASS |
| bsm_eu_call | quantburn-vulkan | quantcandle-cpu | 9.227005508 | 9.227005508 | 0.00e+00 | PASS |
| bsm_eu_put | quantburn-vulkan | quantcandle-cpu | 6.330080628 | 6.330080628 | 0.00e+00 | PASS |
| crr_eu_call | quantburn-vulkan | quantcandle-cpu | 10.43061166 | 10.43061166 | 0.00e+00 | PASS |
| crr_amer_put | quantburn-vulkan | quantcandle-cpu | 6.082354409 | 6.082354409 | 0.00e+00 | PASS |
| amer_put_100 | quantburn-vulkan | quantcandle-cpu | 6.082354409 | 6.082354409 | 0.00e+00 | PASS |
| berm_put | quantburn-vulkan | quantcandle-cpu | 6.025795352 | 6.025795352 | 0.00e+00 | PASS |
| asian_call | quantburn-vulkan | quantcandle-cpu | 6.62561789 | 6.62561789 | 0.00e+00 | PASS(rng) |
| heston_cf_call | quantburn-vulkan | quantcandle-cpu | 9.897022477 | 9.897022477 | 0.00e+00 | PASS |
| heston_mc_call | quantburn-vulkan | quantcandle-cpu | 9.586220714 | 9.586220714 | 0.00e+00 | PASS(rng) |
| sabr_iv | quantburn-vulkan | quantcandle-cpu | 0.0736879502 | 0.0736879502 | 0.00e+00 | PASS |
| sabr_price | quantburn-vulkan | quantcandle-cpu | 0.9367717118 | 0.9367717118 | 0.00e+00 | PASS |
| black76_call | quantburn-vulkan | quantcandle-cpu | 7.577082146 | 7.577082146 | 0.00e+00 | PASS |
| vasicek_zcb_1y | quantburn-vulkan | quantcandle-cpu | 0.9691188083 | 0.9691188083 | 0.00e+00 | PASS |
| vasicek_zcb_5y | quantburn-vulkan | quantcandle-cpu | 0.8396917673 | 0.8396917673 | 0.00e+00 | PASS |
| cir_zcb_1y | quantburn-vulkan | quantcandle-cpu | 0.9690627652 | 0.9690627652 | 0.00e+00 | PASS |
| hw_zcb_1y | quantburn-vulkan | quantcandle-cpu | 0.9704455335 | 0.9704455335 | 0.00e+00 | PASS |
| bk_zcb_5y | quantburn-vulkan | quantcandle-cpu | 0.7105591542 | 0.7105591542 | 0.00e+00 | PASS |
| digital_call | quantburn-vulkan | quantcandle-cpu | 5.796554306 | 5.796554306 | 0.00e+00 | PASS(rng) |
| barrier_uo | quantburn-vulkan | quantcandle-cpu | 1.730618702 | 1.730618702 | 0.00e+00 | PASS(rng) |
| lookback_call | quantburn-vulkan | quantcandle-cpu | 17.65824239 | 17.65824239 | 0.00e+00 | PASS(rng) |
| chooser | quantburn-vulkan | quantcandle-cpu | 6.718645263 | 6.718645263 | 0.00e+00 | PASS |
| bsm_eu_call | quantburn-vulkan | quantcandle-mkl | 9.227005508 | 9.227005508 | 0.00e+00 | PASS |
| bsm_eu_put | quantburn-vulkan | quantcandle-mkl | 6.330080628 | 6.330080628 | 0.00e+00 | PASS |
| crr_eu_call | quantburn-vulkan | quantcandle-mkl | 10.43061166 | 10.43061166 | 0.00e+00 | PASS |
| crr_amer_put | quantburn-vulkan | quantcandle-mkl | 6.082354409 | 6.082354409 | 0.00e+00 | PASS |
| amer_put_100 | quantburn-vulkan | quantcandle-mkl | 6.082354409 | 6.082354409 | 0.00e+00 | PASS |
| berm_put | quantburn-vulkan | quantcandle-mkl | 6.025795352 | 6.025795352 | 0.00e+00 | PASS |
| asian_call | quantburn-vulkan | quantcandle-mkl | 6.62561789 | 6.62561789 | 0.00e+00 | PASS(rng) |
| heston_cf_call | quantburn-vulkan | quantcandle-mkl | 9.897022477 | 9.897022477 | 0.00e+00 | PASS |
| heston_mc_call | quantburn-vulkan | quantcandle-mkl | 9.586220714 | 9.586220714 | 0.00e+00 | PASS(rng) |
| sabr_iv | quantburn-vulkan | quantcandle-mkl | 0.0736879502 | 0.0736879502 | 0.00e+00 | PASS |
| sabr_price | quantburn-vulkan | quantcandle-mkl | 0.9367717118 | 0.9367717118 | 0.00e+00 | PASS |
| black76_call | quantburn-vulkan | quantcandle-mkl | 7.577082146 | 7.577082146 | 0.00e+00 | PASS |
| vasicek_zcb_1y | quantburn-vulkan | quantcandle-mkl | 0.9691188083 | 0.9691188083 | 0.00e+00 | PASS |
| vasicek_zcb_5y | quantburn-vulkan | quantcandle-mkl | 0.8396917673 | 0.8396917673 | 0.00e+00 | PASS |
| cir_zcb_1y | quantburn-vulkan | quantcandle-mkl | 0.9690627652 | 0.9690627652 | 0.00e+00 | PASS |
| hw_zcb_1y | quantburn-vulkan | quantcandle-mkl | 0.9704455335 | 0.9704455335 | 0.00e+00 | PASS |
| bk_zcb_5y | quantburn-vulkan | quantcandle-mkl | 0.7105591542 | 0.7105591542 | 0.00e+00 | PASS |
| digital_call | quantburn-vulkan | quantcandle-mkl | 5.796554306 | 5.796554306 | 0.00e+00 | PASS(rng) |
| barrier_uo | quantburn-vulkan | quantcandle-mkl | 1.730618702 | 1.730618702 | 0.00e+00 | PASS(rng) |
| lookback_call | quantburn-vulkan | quantcandle-mkl | 17.65824239 | 17.65824239 | 0.00e+00 | PASS(rng) |
| chooser | quantburn-vulkan | quantcandle-mkl | 6.718645263 | 6.718645263 | 0.00e+00 | PASS |
| bsm_eu_call | quantburn-vulkan | quantorch-cpu | 9.227005508 | 9.227005508 | 0.00e+00 | PASS |
| bsm_eu_put | quantburn-vulkan | quantorch-cpu | 6.330080628 | 6.330080628 | 0.00e+00 | PASS |
| crr_eu_call | quantburn-vulkan | quantorch-cpu | 10.43061166 | 10.43061166 | 3.07e-15 | PASS |
| crr_amer_put | quantburn-vulkan | quantorch-cpu | 6.082354409 | 6.082354409 | 2.92e-15 | PASS |
| amer_put_100 | quantburn-vulkan | quantorch-cpu | 6.082354409 | 6.082354409 | 2.92e-15 | PASS |
| berm_put | quantburn-vulkan | quantorch-cpu | 6.025795352 | 6.025795352 | 2.36e-15 | PASS |
| asian_call | quantburn-vulkan | quantorch-cpu | 6.62561789 | 7.020513535 | 5.62e-02 | INFO(rng) |
| heston_cf_call | quantburn-vulkan | quantorch-cpu | 9.897022477 | 9.897022247 | 2.32e-08 | PASS(f32) |
| heston_mc_call | quantburn-vulkan | quantorch-cpu | 9.586220714 | 9.927351952 | 3.44e-02 | INFO(rng) |
| sabr_iv | quantburn-vulkan | quantorch-cpu | 0.0736879502 | 0.07368795574 | 7.52e-08 | PASS(f32) |
| sabr_price | quantburn-vulkan | quantorch-cpu | 0.9367717118 | 0.9367747307 | 3.22e-06 | PASS(f32) |
| black76_call | quantburn-vulkan | quantorch-cpu | 7.577082146 | 7.577082146 | 0.00e+00 | PASS |
| vasicek_zcb_1y | quantburn-vulkan | quantorch-cpu | 0.9691188083 | 0.9691188335 | 2.61e-08 | PASS(f32) |
| vasicek_zcb_5y | quantburn-vulkan | quantorch-cpu | 0.8396917673 | 0.8396917582 | 1.09e-08 | PASS(f32) |
| cir_zcb_1y | quantburn-vulkan | quantorch-cpu | 0.9690627652 | 0.9690644145 | 1.70e-06 | PASS(f32) |
| hw_zcb_1y | quantburn-vulkan | quantorch-cpu | 0.9704455335 | 0.9704455137 | 2.04e-08 | PASS(f32) |
| bk_zcb_5y | quantburn-vulkan | quantorch-cpu | 0.7105591542 | 0.7105591893 | 4.95e-08 | PASS(f32) |
| digital_call | quantburn-vulkan | quantorch-cpu | 5.796554306 | 5.164878845 | 1.22e-01 | INFO(rng) |
| barrier_uo | quantburn-vulkan | quantorch-cpu | 1.730618702 | 1.089030385 | 5.89e-01 | INFO(rng) |
| lookback_call | quantburn-vulkan | quantorch-cpu | 17.65824239 | 20.5508728 | 1.41e-01 | INFO(rng) |
| chooser | quantburn-vulkan | quantorch-cpu | 6.718645263 | 6.718645263 | 0.00e+00 | PASS |
| bsm_eu_call | quantburn-vulkan | tfquant-gpu | 9.227005508 | 9.227005508 | 2.70e-15 | PASS |
| bsm_eu_put | quantburn-vulkan | tfquant-gpu | 6.330080628 | 6.330080628 | 0.00e+00 | PASS |
| crr_eu_call | quantburn-vulkan | tfquant-gpu | 10.43061166 | 10.43061166 | 5.65e-14 | PASS |
| crr_amer_put | quantburn-vulkan | tfquant-gpu | 6.082354409 | 6.082354409 | 1.39e-13 | PASS |
| amer_put_100 | quantburn-vulkan | tfquant-gpu | 6.082354409 | 6.082354409 | 1.39e-13 | PASS |
| bsm_eu_call | quantcandle-cpu | quantcandle-mkl | 9.227005508 | 9.227005508 | 0.00e+00 | PASS |
| bsm_eu_put | quantcandle-cpu | quantcandle-mkl | 6.330080628 | 6.330080628 | 0.00e+00 | PASS |
| crr_eu_call | quantcandle-cpu | quantcandle-mkl | 10.43061166 | 10.43061166 | 0.00e+00 | PASS |
| crr_amer_put | quantcandle-cpu | quantcandle-mkl | 6.082354409 | 6.082354409 | 0.00e+00 | PASS |
| amer_put_100 | quantcandle-cpu | quantcandle-mkl | 6.082354409 | 6.082354409 | 0.00e+00 | PASS |
| berm_put | quantcandle-cpu | quantcandle-mkl | 6.025795352 | 6.025795352 | 0.00e+00 | PASS |
| asian_call | quantcandle-cpu | quantcandle-mkl | 6.62561789 | 6.62561789 | 0.00e+00 | PASS(rng) |
| heston_cf_call | quantcandle-cpu | quantcandle-mkl | 9.897022477 | 9.897022477 | 0.00e+00 | PASS |
| heston_mc_call | quantcandle-cpu | quantcandle-mkl | 9.586220714 | 9.586220714 | 0.00e+00 | PASS(rng) |
| sabr_iv | quantcandle-cpu | quantcandle-mkl | 0.0736879502 | 0.0736879502 | 0.00e+00 | PASS |
| sabr_price | quantcandle-cpu | quantcandle-mkl | 0.9367717118 | 0.9367717118 | 0.00e+00 | PASS |
| black76_call | quantcandle-cpu | quantcandle-mkl | 7.577082146 | 7.577082146 | 0.00e+00 | PASS |
| vasicek_zcb_1y | quantcandle-cpu | quantcandle-mkl | 0.9691188083 | 0.9691188083 | 0.00e+00 | PASS |
| vasicek_zcb_5y | quantcandle-cpu | quantcandle-mkl | 0.8396917673 | 0.8396917673 | 0.00e+00 | PASS |
| cir_zcb_1y | quantcandle-cpu | quantcandle-mkl | 0.9690627652 | 0.9690627652 | 0.00e+00 | PASS |
| hw_zcb_1y | quantcandle-cpu | quantcandle-mkl | 0.9704455335 | 0.9704455335 | 0.00e+00 | PASS |
| bk_zcb_5y | quantcandle-cpu | quantcandle-mkl | 0.7105591542 | 0.7105591542 | 0.00e+00 | PASS |
| digital_call | quantcandle-cpu | quantcandle-mkl | 5.796554306 | 5.796554306 | 0.00e+00 | PASS(rng) |
| barrier_uo | quantcandle-cpu | quantcandle-mkl | 1.730618702 | 1.730618702 | 0.00e+00 | PASS(rng) |
| lookback_call | quantcandle-cpu | quantcandle-mkl | 17.65824239 | 17.65824239 | 0.00e+00 | PASS(rng) |
| chooser | quantcandle-cpu | quantcandle-mkl | 6.718645263 | 6.718645263 | 0.00e+00 | PASS |
| bsm_eu_call | quantcandle-cpu | quantorch-cpu | 9.227005508 | 9.227005508 | 0.00e+00 | PASS |
| bsm_eu_put | quantcandle-cpu | quantorch-cpu | 6.330080628 | 6.330080628 | 0.00e+00 | PASS |
| crr_eu_call | quantcandle-cpu | quantorch-cpu | 10.43061166 | 10.43061166 | 3.07e-15 | PASS |
| crr_amer_put | quantcandle-cpu | quantorch-cpu | 6.082354409 | 6.082354409 | 2.92e-15 | PASS |
| amer_put_100 | quantcandle-cpu | quantorch-cpu | 6.082354409 | 6.082354409 | 2.92e-15 | PASS |
| berm_put | quantcandle-cpu | quantorch-cpu | 6.025795352 | 6.025795352 | 2.36e-15 | PASS |
| asian_call | quantcandle-cpu | quantorch-cpu | 6.62561789 | 7.020513535 | 5.62e-02 | INFO(rng) |
| heston_cf_call | quantcandle-cpu | quantorch-cpu | 9.897022477 | 9.897022247 | 2.32e-08 | PASS(f32) |
| heston_mc_call | quantcandle-cpu | quantorch-cpu | 9.586220714 | 9.927351952 | 3.44e-02 | INFO(rng) |
| sabr_iv | quantcandle-cpu | quantorch-cpu | 0.0736879502 | 0.07368795574 | 7.52e-08 | PASS(f32) |
| sabr_price | quantcandle-cpu | quantorch-cpu | 0.9367717118 | 0.9367747307 | 3.22e-06 | PASS(f32) |
| black76_call | quantcandle-cpu | quantorch-cpu | 7.577082146 | 7.577082146 | 0.00e+00 | PASS |
| vasicek_zcb_1y | quantcandle-cpu | quantorch-cpu | 0.9691188083 | 0.9691188335 | 2.61e-08 | PASS(f32) |
| vasicek_zcb_5y | quantcandle-cpu | quantorch-cpu | 0.8396917673 | 0.8396917582 | 1.09e-08 | PASS(f32) |
| cir_zcb_1y | quantcandle-cpu | quantorch-cpu | 0.9690627652 | 0.9690644145 | 1.70e-06 | PASS(f32) |
| hw_zcb_1y | quantcandle-cpu | quantorch-cpu | 0.9704455335 | 0.9704455137 | 2.04e-08 | PASS(f32) |
| bk_zcb_5y | quantcandle-cpu | quantorch-cpu | 0.7105591542 | 0.7105591893 | 4.95e-08 | PASS(f32) |
| digital_call | quantcandle-cpu | quantorch-cpu | 5.796554306 | 5.164878845 | 1.22e-01 | INFO(rng) |
| barrier_uo | quantcandle-cpu | quantorch-cpu | 1.730618702 | 1.089030385 | 5.89e-01 | INFO(rng) |
| lookback_call | quantcandle-cpu | quantorch-cpu | 17.65824239 | 20.5508728 | 1.41e-01 | INFO(rng) |
| chooser | quantcandle-cpu | quantorch-cpu | 6.718645263 | 6.718645263 | 0.00e+00 | PASS |
| bsm_eu_call | quantcandle-cpu | tfquant-gpu | 9.227005508 | 9.227005508 | 2.70e-15 | PASS |
| bsm_eu_put | quantcandle-cpu | tfquant-gpu | 6.330080628 | 6.330080628 | 0.00e+00 | PASS |
| crr_eu_call | quantcandle-cpu | tfquant-gpu | 10.43061166 | 10.43061166 | 5.65e-14 | PASS |
| crr_amer_put | quantcandle-cpu | tfquant-gpu | 6.082354409 | 6.082354409 | 1.39e-13 | PASS |
| amer_put_100 | quantcandle-cpu | tfquant-gpu | 6.082354409 | 6.082354409 | 1.39e-13 | PASS |
| bsm_eu_call | quantcandle-mkl | quantorch-cpu | 9.227005508 | 9.227005508 | 0.00e+00 | PASS |
| bsm_eu_put | quantcandle-mkl | quantorch-cpu | 6.330080628 | 6.330080628 | 0.00e+00 | PASS |
| crr_eu_call | quantcandle-mkl | quantorch-cpu | 10.43061166 | 10.43061166 | 3.07e-15 | PASS |
| crr_amer_put | quantcandle-mkl | quantorch-cpu | 6.082354409 | 6.082354409 | 2.92e-15 | PASS |
| amer_put_100 | quantcandle-mkl | quantorch-cpu | 6.082354409 | 6.082354409 | 2.92e-15 | PASS |
| berm_put | quantcandle-mkl | quantorch-cpu | 6.025795352 | 6.025795352 | 2.36e-15 | PASS |
| asian_call | quantcandle-mkl | quantorch-cpu | 6.62561789 | 7.020513535 | 5.62e-02 | INFO(rng) |
| heston_cf_call | quantcandle-mkl | quantorch-cpu | 9.897022477 | 9.897022247 | 2.32e-08 | PASS(f32) |
| heston_mc_call | quantcandle-mkl | quantorch-cpu | 9.586220714 | 9.927351952 | 3.44e-02 | INFO(rng) |
| sabr_iv | quantcandle-mkl | quantorch-cpu | 0.0736879502 | 0.07368795574 | 7.52e-08 | PASS(f32) |
| sabr_price | quantcandle-mkl | quantorch-cpu | 0.9367717118 | 0.9367747307 | 3.22e-06 | PASS(f32) |
| black76_call | quantcandle-mkl | quantorch-cpu | 7.577082146 | 7.577082146 | 0.00e+00 | PASS |
| vasicek_zcb_1y | quantcandle-mkl | quantorch-cpu | 0.9691188083 | 0.9691188335 | 2.61e-08 | PASS(f32) |
| vasicek_zcb_5y | quantcandle-mkl | quantorch-cpu | 0.8396917673 | 0.8396917582 | 1.09e-08 | PASS(f32) |
| cir_zcb_1y | quantcandle-mkl | quantorch-cpu | 0.9690627652 | 0.9690644145 | 1.70e-06 | PASS(f32) |
| hw_zcb_1y | quantcandle-mkl | quantorch-cpu | 0.9704455335 | 0.9704455137 | 2.04e-08 | PASS(f32) |
| bk_zcb_5y | quantcandle-mkl | quantorch-cpu | 0.7105591542 | 0.7105591893 | 4.95e-08 | PASS(f32) |
| digital_call | quantcandle-mkl | quantorch-cpu | 5.796554306 | 5.164878845 | 1.22e-01 | INFO(rng) |
| barrier_uo | quantcandle-mkl | quantorch-cpu | 1.730618702 | 1.089030385 | 5.89e-01 | INFO(rng) |
| lookback_call | quantcandle-mkl | quantorch-cpu | 17.65824239 | 20.5508728 | 1.41e-01 | INFO(rng) |
| chooser | quantcandle-mkl | quantorch-cpu | 6.718645263 | 6.718645263 | 0.00e+00 | PASS |
| bsm_eu_call | quantcandle-mkl | tfquant-gpu | 9.227005508 | 9.227005508 | 2.70e-15 | PASS |
| bsm_eu_put | quantcandle-mkl | tfquant-gpu | 6.330080628 | 6.330080628 | 0.00e+00 | PASS |
| crr_eu_call | quantcandle-mkl | tfquant-gpu | 10.43061166 | 10.43061166 | 5.65e-14 | PASS |
| crr_amer_put | quantcandle-mkl | tfquant-gpu | 6.082354409 | 6.082354409 | 1.39e-13 | PASS |
| amer_put_100 | quantcandle-mkl | tfquant-gpu | 6.082354409 | 6.082354409 | 1.39e-13 | PASS |
| bsm_eu_call | quantorch-cpu | tfquant-gpu | 9.227005508 | 9.227005508 | 2.70e-15 | PASS |
| bsm_eu_put | quantorch-cpu | tfquant-gpu | 6.330080628 | 6.330080628 | 0.00e+00 | PASS |
| crr_eu_call | quantorch-cpu | tfquant-gpu | 10.43061166 | 10.43061166 | 5.35e-14 | PASS |
| crr_amer_put | quantorch-cpu | tfquant-gpu | 6.082354409 | 6.082354409 | 1.36e-13 | PASS |
| amer_put_100 | quantorch-cpu | tfquant-gpu | 6.082354409 | 6.082354409 | 1.36e-13 | PASS |
