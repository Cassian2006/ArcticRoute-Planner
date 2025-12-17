import arcticroute
import arcticroute.core.eco as eco
print("eco module file:", eco.__file__)
print("eco module attrs:", [a for a in dir(eco) if a in ("fuel_per_nm_map","eco_cost_norm","eval_route_eco")])

