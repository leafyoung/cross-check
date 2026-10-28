# Invariant check results

- quantburn I1 put-call parity: C-P=2.896924880604 vs 2.896924880604 err=0.0e+00 OK
- quantburn I4 CRR(500)->BSM(q=0): 10.4465851364 vs 10.4505835722 rel=3.8e-04 OK
- quantcandle I1 put-call parity: C-P=2.896924880604 vs 2.896924880604 err=0.0e+00 OK
- quantcandle I4 CRR(500)->BSM(q=0): 10.4465851364 vs 10.4505835722 rel=3.8e-04 OK
- quantjax I1 put-call parity: C-P=2.896924880604 vs 2.896924880604 err=0.0e+00 OK
- quantjax I4 CRR(500)->BSM(q=0): 10.4465851364 vs 10.4505835722 rel=3.8e-04 OK
- tfquant I1 put-call parity: err=1.8e-14 OK
- tfquant I4 CRR(500)->BSM: rel=3.8e-04 OK
- quantjax I2 AoN-K*CoN=vanilla: 9.227005508154 vs 9.227005508154 err=3.6e-14 OK
- quantjax I3 barrier in+out=vanilla: 9.227005508154 vs 9.227005508154 err=0.0e+00 OK
- quantburn I5 ZCB monotone decreasing: ['0.975318', '0.951286', '0.905225', '0.782649', '0.621866'] OK

**Violations: 0** — all invariants hold
