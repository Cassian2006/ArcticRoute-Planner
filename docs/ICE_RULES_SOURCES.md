

## Phase 14 – Authoritative thresholds pinned (POLARIS + Polar Code)

### POLARIS (IMO MSC.1/Circ.1519)
- RIO → Operation level thresholds (Table 1.1)
  - normal: RIO >= 0
  - elevated: -10 <= RIO < 0
  - special consideration: RIO < -10
  - below PC7 / no ice class: elevated risk is treated as special consideration
- Speed limitations in elevated risk (Table 1.2)
  - PC1: 11 kn, PC2: 8 kn, PC3–PC5: 5 kn, below PC5: 3 kn
- Decayed ice (Table 1.4) usage
  - default uses standard table (Table 1.3); only enable decayed table when confirmed "decayed ice" per operational assessment

### Polar Code (IMO MSC.385(94)) – semantics for documentation
- Open water: sea ice concentration < 1/10 (0.1)
- FYI thickness ranges used for interpretation:
  - thin FYI: 0.3–0.7 m
  - medium FYI: 0.7–1.2 m
  - FYI general: 0.3–2.0 m

### Notes (ship-specific; keep configurable, do not claim universal)
- SWH / wave thresholds and speed reduction rules are ship/PWOM dependent
- fuel model parameters are ship-specific
