import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2, ceil
from sklearn.cluster import KMeans
import json
from pathlib import Path

# =========================================================
# 1. 基础数据
# =========================================================
STORE_DATA = {
    "store_id": list(range(1, 22)),
    "longitude": [
        113.1624, 113.1676, 113.1099, 113.1377, 113.1398,
        113.1767, 113.2750, 112.9989, 113.1168, 113.1988,
        113.1814, 113.1023, 113.1122, 113.1243, 113.0141,
        113.2464, 113.0457, 113.1311, 113.2188, 113.3031,
        113.1982
    ],
    "latitude": [
        23.03142, 23.05064, 23.03248, 22.99512, 23.04658,
        23.11890, 22.75460, 23.11711, 23.00881, 23.11656,
        23.08196, 23.00104, 22.94741, 23.09788, 22.95301,
        22.83936, 23.05123, 23.01737, 22.86602, 22.77021,
        23.02934
    ]
}
df_stores = pd.DataFrame(STORE_DATA)

# =========================================================
# 2. 参数设置
# =========================================================
MONTHS = list(range(1, 13))
DAYS_PER_YEAR = 365
BASE_DEMAND_START = 50
BASE_DEMAND_END = 120

CAPACITY_PER_EQUIPMENT = 500
EQUIPMENT_COST = 80000
EQUIPMENT_LIFE_YEARS = 3
SERVICE_LEVEL = 0.95
SAFETY_MARGIN = 0.12
MAX_EQUIPMENT_PER_FACILITY = 4

MAX_ROUTE_HOURS = 3.0
BOUNDARY_THRESHOLD_KM = 8

LATE_PENALTY = 10000
SHORTAGE_PENALTY = 12000
DISPATCH_OVERLOAD_PENALTY = 15000

MAX_VEHICLES_PER_FACILITY = {
    "sanlun": 2,
    "jinbei": 1,
    "rider": 2
}

VEHICLES = {
    "sanlun": {
        "capacity": 500,
        "speed_kmh": 18,
        "cost_func": lambda stops: 170 + 30 * stops
    },
    "jinbei": {
        "capacity": 1000,
        "speed_kmh": 28,
        "cost_func": lambda stops: 280 + 30 * stops
    },
    "rider": {
        "capacity": 50,
        "speed_kmh": 12,
        "hourly_cost": 70
    }
}

# =========================================================
# 3. 工具函数
# =========================================================
def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def calc_daily_rent(equipment_num):
    if equipment_num <= 0:
        return 0
    if equipment_num == 1:
        return 50
    return 50 + (equipment_num - 1) * 40

def compute_route_distance_by_ids(df, facility_id, route_ids):
    facility_row = df[df["store_id"] == facility_id].iloc[0]
    proc_lon, proc_lat = facility_row["longitude"], facility_row["latitude"]

    ordered_rows = [df[df["store_id"] == sid].iloc[0] for sid in route_ids]

    total = 0.0
    prev_lon, prev_lat = proc_lon, proc_lat
    for row in ordered_rows:
        total += haversine_km(prev_lon, prev_lat, row["longitude"], row["latitude"])
        prev_lon, prev_lat = row["longitude"], row["latitude"]

    # 返程
    total += haversine_km(prev_lon, prev_lat, proc_lon, proc_lat)
    return total

def two_opt_route(df, facility_id, route_ids):
    if len(route_ids) <= 2:
        return route_ids[:]

    best_route = route_ids[:]
    best_distance = compute_route_distance_by_ids(df, facility_id, best_route)

    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                new_route = best_route[:]
                new_route[i:j] = reversed(new_route[i:j])
                new_distance = compute_route_distance_by_ids(df, facility_id, new_route)
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improved = True
    return best_route

# =========================================================
# 4. 多周期需求与场景生成
# =========================================================
def generate_base_monthly_demand():
    return {
        t: BASE_DEMAND_START + (BASE_DEMAND_END - BASE_DEMAND_START) * (t - 1) / 11
        for t in MONTHS
    }

def generate_store_factors(df, seed=42):
    rng = np.random.default_rng(seed)
    factors = rng.uniform(0.82, 1.18, size=len(df))
    return dict(zip(df["store_id"], factors))

def generate_scenarios(df, n_scenarios=300, seed=42):
    rng = np.random.default_rng(seed)
    base_monthly = generate_base_monthly_demand()
    store_factors = generate_store_factors(df, seed=seed)

    scenarios = []
    for s in range(n_scenarios):
        demand_dict = {}
        travel_factor = {}
        service_time = {}

        for month in MONTHS:
            month_std = 0.10 if month <= 8 else 0.20
            for sid in df["store_id"]:
                xi = np.clip(rng.normal(1.0, month_std), 0.70, 1.45)
                demand = base_monthly[month] * store_factors[sid] * xi
                demand_dict[(sid, month)] = max(1, demand)

        for sid in df["store_id"]:
            service_time[sid] = rng.uniform(6, 15) / 60.0  # 小时

        for i in df["store_id"]:
            for j in df["store_id"]:
                if i != j:
                    tf = rng.choice([1.0, 1.2, 1.4, 1.7, 2.0], p=[0.35, 0.25, 0.20, 0.15, 0.05])
                    travel_factor[(i, j)] = tf

        scenarios.append({
            "scenario_id": s,
            "demand": demand_dict,
            "travel_factor": travel_factor,
            "service_time": service_time
        })

    return scenarios

# =========================================================
# 5. 候选选址方案生成
# =========================================================
def generate_candidate_facility_schemes(df, k_list=(2, 3, 4, 5), random_state=42):
    X = df[["longitude", "latitude"]].values
    schemes = []

    for k in k_list:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_

        temp = df.copy()
        temp["cluster"] = labels
        facilities = []

        for c in range(k):
            cluster_df = temp[temp["cluster"] == c].copy()
            center_lon, center_lat = centers[c]
            cluster_df["dist_to_center"] = cluster_df.apply(
                lambda r: np.sqrt((r["longitude"] - center_lon) ** 2 + (r["latitude"] - center_lat) ** 2),
                axis=1
            )
            facility = int(cluster_df.loc[cluster_df["dist_to_center"].idxmin(), "store_id"])
            facilities.append(facility)

        schemes.append({"k": k, "facility_ids": facilities})

    return schemes

# =========================================================
# 6. 主服务 + 备选服务分配
# =========================================================
def assign_stores_with_backup(df, facility_ids):
    facility_df = df[df["store_id"].isin(facility_ids)].copy()
    rows = []

    for _, row in df.iterrows():
        sid = int(row["store_id"])
        distances = []

        for _, frow in facility_df.iterrows():
            fid = int(frow["store_id"])
            dist = haversine_km(row["longitude"], row["latitude"], frow["longitude"], frow["latitude"])
            distances.append((fid, dist))

        distances = sorted(distances, key=lambda x: x[1])
        primary = distances[0][0]
        backup = distances[1][0] if len(distances) > 1 else primary

        rows.append({
            "store_id": sid,
            "primary_facility": primary,
            "backup_facility": backup,
            "dist_primary": distances[0][1],
            "dist_backup": distances[1][1] if len(distances) > 1 else distances[0][1],
            "is_boundary": (distances[1][1] - distances[0][1] <= BOUNDARY_THRESHOLD_KM) if len(distances) > 1 else False
        })

    return pd.DataFrame(rows)

# =========================================================
# 7. 设备配置
# =========================================================
def configure_equipment(df_assign, scenarios, facility_ids):
    rows = []

    for fid in facility_ids:
        served = df_assign[df_assign["primary_facility"] == fid]["store_id"].tolist()
        monthly_loads = []

        for sc in scenarios:
            for month in MONTHS:
                monthly_loads.append(sum(sc["demand"][(sid, month)] for sid in served))

        robust_load = np.quantile(monthly_loads, SERVICE_LEVEL) * (1 + SAFETY_MARGIN)
        equipment_num = ceil(robust_load / CAPACITY_PER_EQUIPMENT)
        equipment_num = min(equipment_num, MAX_EQUIPMENT_PER_FACILITY)

        rows.append({
            "facility_id": fid,
            "robust_load": robust_load,
            "equipment_num": equipment_num,
            "capacity_limit": equipment_num * CAPACITY_PER_EQUIPMENT
        })

    return pd.DataFrame(rows)

# =========================================================
# 8. 边界门店动态切换
# =========================================================
def dynamic_reassign_boundaries(df_assign, equipment_df, demand_per_store):
    current_assign = df_assign.copy()

    def get_load(fid):
        stores = current_assign[current_assign["primary_facility"] == fid]["store_id"].tolist()
        return sum(demand_per_store[sid] for sid in stores)

    improved = True
    while improved:
        improved = False
        boundary_df = current_assign[current_assign["is_boundary"]].copy()

        for _, row in boundary_df.iterrows():
            sid = int(row["store_id"])
            primary = int(current_assign.loc[current_assign["store_id"] == sid, "primary_facility"].iloc[0])
            backup = int(current_assign.loc[current_assign["store_id"] == sid, "backup_facility"].iloc[0])

            p_cap = float(equipment_df[equipment_df["facility_id"] == primary]["capacity_limit"].iloc[0])
            b_cap = float(equipment_df[equipment_df["facility_id"] == backup]["capacity_limit"].iloc[0])

            p_load = get_load(primary)
            b_load = get_load(backup)

            if p_load > 0.92 * p_cap and (b_load + demand_per_store[sid] <= b_cap):
                current_assign.loc[current_assign["store_id"] == sid, "primary_facility"] = backup
                improved = True

    return current_assign

# =========================================================
# 9. Clarke-Wright 节约算法
# =========================================================
def clarke_wright_routes(df, facility_id, assigned_store_ids, demand_per_store, vehicle_capacity):
    """
    输入：
    - facility_id: 加工店
    - assigned_store_ids: 该加工店服务的门店（包含facility_id本身也无妨）
    输出：
    - 若干条路径，每条路径是 store_id 列表（不含 facility_id）
    """
    customers = [sid for sid in assigned_store_ids if sid != facility_id]
    if len(customers) == 0:
        return []

    # 初始：每个客户单独一路
    routes = {sid: [sid] for sid in customers}
    route_load = {sid: demand_per_store[sid] for sid in customers}

    def route_distance(route):
        return compute_route_distance_by_ids(df, facility_id, route)

    # 节约值
    savings = []
    facility_row = df[df["store_id"] == facility_id].iloc[0]
    flon, flat = facility_row["longitude"], facility_row["latitude"]

    for i in customers:
        irow = df[df["store_id"] == i].iloc[0]
        d0i = haversine_km(flon, flat, irow["longitude"], irow["latitude"])
        for j in customers:
            if i < j:
                jrow = df[df["store_id"] == j].iloc[0]
                d0j = haversine_km(flon, flat, jrow["longitude"], jrow["latitude"])
                dij = haversine_km(irow["longitude"], irow["latitude"], jrow["longitude"], jrow["latitude"])
                s = d0i + d0j - dij
                savings.append((i, j, s))

    savings = sorted(savings, key=lambda x: x[2], reverse=True)

    # 记录每个客户当前属于哪条路线
    belongs = {sid: sid for sid in customers}

    for i, j, s in savings:
        ri_key = belongs[i]
        rj_key = belongs[j]

        if ri_key == rj_key:
            continue

        route_i = routes[ri_key]
        route_j = routes[rj_key]

        # Clarke-Wright 标准端点合并：只允许端点接端点
        candidates = []

        # i在route_i左端, j在route_j右端
        if route_i[0] == i and route_j[-1] == j:
            merged = route_j + route_i
            candidates.append(merged)

        # i在route_i右端, j在route_j左端
        if route_i[-1] == i and route_j[0] == j:
            merged = route_i + route_j
            candidates.append(merged)

        # i在route_i左端, j在route_j左端
        if route_i[0] == i and route_j[0] == j:
            merged = list(reversed(route_i)) + route_j
            candidates.append(merged)

        # i在route_i右端, j在route_j右端
        if route_i[-1] == i and route_j[-1] == j:
            merged = route_i + list(reversed(route_j))
            candidates.append(merged)

        if not candidates:
            continue

        merged_ok = None
        best_distance = None

        for cand in candidates:
            load = sum(demand_per_store[x] for x in cand)
            if load > vehicle_capacity:
                continue

            dist = route_distance(cand)
            if best_distance is None or dist < best_distance:
                best_distance = dist
                merged_ok = cand

        if merged_ok is None:
            continue

        # 合并成功，更新
        new_key = min(ri_key, rj_key)
        old_key = max(ri_key, rj_key)

        routes[new_key] = merged_ok
        route_load[new_key] = sum(demand_per_store[x] for x in merged_ok)

        for sid in merged_ok:
            belongs[sid] = new_key

        if old_key in routes:
            del routes[old_key]
        if old_key in route_load:
            del route_load[old_key]

    return list(routes.values())

# =========================================================
# 10. 车型选择与线路评价
# =========================================================
def evaluate_route(df, route_ids, facility_id, scenario, demand_per_store):
    facility_row = df[df["store_id"] == facility_id].iloc[0]
    proc_lon, proc_lat = facility_row["longitude"], facility_row["latitude"]

    total_demand = sum(demand_per_store[sid] for sid in route_ids)
    num_stops = len(route_ids)
    ordered_rows = [df[df["store_id"] == sid].iloc[0] for sid in route_ids]

    best_option = None

    for vehicle_name, info in VEHICLES.items():
        if total_demand > info["capacity"]:
            continue

        total_time = 0.0
        total_distance = 0.0

        prev_id = facility_id
        prev_lon, prev_lat = proc_lon, proc_lat

        for row in ordered_rows:
            sid = int(row["store_id"])
            leg_dist = haversine_km(prev_lon, prev_lat, row["longitude"], row["latitude"])
            tf = scenario["travel_factor"].get((prev_id, sid), 1.0)

            total_distance += leg_dist
            total_time += leg_dist / info["speed_kmh"] * tf
            total_time += scenario["service_time"][sid]

            prev_id = sid
            prev_lon, prev_lat = row["longitude"], row["latitude"]

        # 返程
        back_dist = haversine_km(prev_lon, prev_lat, proc_lon, proc_lat)
        tf_back = scenario["travel_factor"].get((prev_id, facility_id), 1.0)
        total_distance += back_dist
        total_time += back_dist / info["speed_kmh"] * tf_back

        if vehicle_name == "rider":
            route_cost = total_time * info["hourly_cost"]
        else:
            route_cost = info["cost_func"](num_stops)

        option = {
            "vehicle_type": vehicle_name,
            "route_cost": route_cost,
            "route_time": total_time,
            "route_distance": total_distance,
            "feasible": total_time <= MAX_ROUTE_HOURS
        }

        if best_option is None:
            best_option = option
        else:
            if option["feasible"] and not best_option["feasible"]:
                best_option = option
            elif option["feasible"] == best_option["feasible"] and option["route_cost"] < best_option["route_cost"]:
                best_option = option

    return best_option

# =========================================================
# 11. 调度可行性检查
# =========================================================
def check_dispatch_feasibility(route_results):
    summary = {}
    overload = False

    for rt in route_results:
        vtype = rt["vehicle_type"]
        summary.setdefault(vtype, 0.0)
        summary[vtype] += rt["route_time"]

    for vtype, total_hours in summary.items():
        max_hours = MAX_VEHICLES_PER_FACILITY[vtype] * MAX_ROUTE_HOURS
        if total_hours > max_hours:
            overload = True

    return overload, summary

# =========================================================
# 12. 仿真方案
# =========================================================
def simulate_scheme(df, scheme, scenarios):
    facility_ids = scheme["facility_ids"]

    df_assign = assign_stores_with_backup(df, facility_ids)
    equipment_df = configure_equipment(df_assign, scenarios, facility_ids)

    # 固定成本
    fixed_rows = []
    for _, row in equipment_df.iterrows():
        fid = int(row["facility_id"])
        eq = int(row["equipment_num"])
        annual_rent = calc_daily_rent(eq) * DAYS_PER_YEAR
        annual_equipment_cost = eq * EQUIPMENT_COST / EQUIPMENT_LIFE_YEARS
        fixed_rows.append({
            "facility_id": fid,
            "equipment_num": eq,
            "annual_rent": annual_rent,
            "annual_equipment_cost": annual_equipment_cost,
            "annual_fixed_cost": annual_rent + annual_equipment_cost
        })

    fixed_df = pd.DataFrame(fixed_rows)
    total_fixed_cost = fixed_df["annual_fixed_cost"].sum()

    scenario_rows = []

    for sc in scenarios:
        demand_per_store = {sid: sc["demand"][(sid, 12)] for sid in df["store_id"]}

        current_assign = dynamic_reassign_boundaries(df_assign, equipment_df, demand_per_store)

        transport_cost_day = 0.0
        late_count = 0
        shortage_count = 0
        overload_count = 0
        total_routes = 0
        max_equipment_util = 0.0

        for fid in facility_ids:
            served = current_assign[current_assign["primary_facility"] == fid]["store_id"].tolist()

            current_load = sum(demand_per_store[sid] for sid in served)
            cap_limit = float(equipment_df[equipment_df["facility_id"] == fid]["capacity_limit"].iloc[0])

            util = current_load / cap_limit if cap_limit > 0 else 0
            max_equipment_util = max(max_equipment_util, util)

            if current_load > cap_limit:
                shortage_count += 1
                transport_cost_day += SHORTAGE_PENALTY

            # 先用三轮容量构造Clarke-Wright路线
            routes = clarke_wright_routes(
                df, fid, served, demand_per_store, vehicle_capacity=VEHICLES["sanlun"]["capacity"]
            )

            facility_route_results = []

            for route in routes:
                if len(route) == 0:
                    continue

                # 再做2-opt精修
                if len(route) >= 3:
                    route = two_opt_route(df, fid, route)

                result = evaluate_route(df, route, fid, sc, demand_per_store)

                if result is None:
                    # 仍不可行则拆成单点应急
                    for sid in route:
                        single_result = evaluate_route(df, [sid], fid, sc, demand_per_store)
                        total_routes += 1
                        if single_result is None:
                            shortage_count += 1
                            late_count += 1
                            transport_cost_day += SHORTAGE_PENALTY + LATE_PENALTY
                        else:
                            transport_cost_day += single_result["route_cost"]
                            if not single_result["feasible"]:
                                late_count += 1
                                transport_cost_day += LATE_PENALTY
                            facility_route_results.append(single_result)
                    continue

                total_routes += 1
                transport_cost_day += result["route_cost"]

                if not result["feasible"]:
                    late_count += 1
                    transport_cost_day += LATE_PENALTY

                facility_route_results.append(result)

            overload, dispatch_summary = check_dispatch_feasibility(facility_route_results)
            if overload:
                overload_count += 1
                transport_cost_day += DISPATCH_OVERLOAD_PENALTY

        annual_transport_cost = transport_cost_day * DAYS_PER_YEAR
        annual_total_cost = annual_transport_cost + total_fixed_cost

        scenario_rows.append({
            "scenario_id": sc["scenario_id"],
            "annual_transport_cost": annual_transport_cost,
            "annual_total_cost": annual_total_cost,
            "late_count": late_count,
            "shortage_count": shortage_count,
            "overload_count": overload_count,
            "total_routes": total_routes,
            "max_equipment_util": max_equipment_util
        })

    scenario_df = pd.DataFrame(scenario_rows)

    mean_total_cost = scenario_df["annual_total_cost"].mean()
    p95_total_cost = scenario_df["annual_total_cost"].quantile(0.95)
    worst_total_cost = scenario_df["annual_total_cost"].max()
    threshold = scenario_df["annual_total_cost"].quantile(0.95)
    cvar95 = scenario_df[scenario_df["annual_total_cost"] >= threshold]["annual_total_cost"].mean()

    late_rate = (scenario_df["late_count"] > 0).mean()
    shortage_rate = (scenario_df["shortage_count"] > 0).mean()
    overload_rate = (scenario_df["overload_count"] > 0).mean()
    avg_max_equipment_util = scenario_df["max_equipment_util"].mean()

    avg_load_share = []
    for fid in facility_ids:
        served_num = df_assign[df_assign["primary_facility"] == fid]["store_id"].nunique()
        avg_load_share.append(served_num / len(df))
    load_concentration = max(avg_load_share)

    summary = {
        "k": scheme["k"],
        "facility_ids": facility_ids,
        "total_fixed_cost": total_fixed_cost,
        "mean_total_cost": mean_total_cost,
        "p95_total_cost": p95_total_cost,
        "worst_total_cost": worst_total_cost,
        "cvar95": cvar95,
        "late_rate": late_rate,
        "shortage_rate": shortage_rate,
        "overload_rate": overload_rate,
        "avg_max_equipment_util": avg_max_equipment_util,
        "load_concentration": load_concentration
    }

    return summary, fixed_df, scenario_df, df_assign, equipment_df

# =========================================================
# 13. 鲁棒评分
# =========================================================
def robust_score(summary):
    return (
        summary["mean_total_cost"]
        + 0.4 * summary["cvar95"]
        + 80000 * summary["late_rate"]
        + 100000 * summary["shortage_rate"]
        + 70000 * summary["overload_rate"]
        + 120000 * max(0, summary["avg_max_equipment_util"] - 0.90)
        + 60000 * max(0, summary["load_concentration"] - 0.55)
    )

def build_representative_scenario(df, scenarios, month=12):
    service_time = {}
    travel_factor = {}
    demand_per_store = {}

    for sid in df["store_id"]:
        service_time[sid] = float(np.mean([sc["service_time"][sid] for sc in scenarios]))
        demand_per_store[sid] = float(np.mean([sc["demand"][(sid, month)] for sc in scenarios]))

    for i in df["store_id"]:
        for j in df["store_id"]:
            if i == j:
                continue
            travel_factor[(i, j)] = float(np.mean([sc["travel_factor"].get((i, j), 1.0) for sc in scenarios]))

    return {
        "scenario_id": "representative",
        "service_time": service_time,
        "travel_factor": travel_factor,
        "demand": {(sid, month): demand_per_store[sid] for sid in df["store_id"]},
        "demand_per_store": demand_per_store
    }

def build_representative_route_df(df, assign_df, equipment_df, facility_ids, scenarios, month=12):
    representative = build_representative_scenario(df, scenarios, month=month)
    demand_per_store = representative["demand_per_store"]
    current_assign = dynamic_reassign_boundaries(assign_df, equipment_df, demand_per_store)

    route_rows = []
    route_id = 1

    for fid in facility_ids:
        served = current_assign[current_assign["primary_facility"] == fid]["store_id"].tolist()

        routes = clarke_wright_routes(
            df,
            fid,
            served,
            demand_per_store,
            vehicle_capacity=VEHICLES["sanlun"]["capacity"]
        )

        for route in routes:
            if len(route) == 0:
                continue

            if len(route) >= 3:
                route = two_opt_route(df, fid, route)

            result = evaluate_route(df, route, fid, representative, demand_per_store)
            route_kind = "direct"
            split_routes = [route]

            if result is None:
                route_kind = "fallback_single"
                split_routes = [[sid] for sid in route]

            for split_route in split_routes:
                split_result = evaluate_route(df, split_route, fid, representative, demand_per_store)
                current_route_id = route_id
                route_id += 1

                for seq, sid in enumerate(split_route, start=1):
                    row = {
                        "scenario_id": representative["scenario_id"],
                        "facility_id": fid,
                        "route_id": current_route_id,
                        "seq": seq,
                        "store_id": sid,
                        "store_demand": demand_per_store[sid],
                        "route_kind": route_kind
                    }

                    if split_result is not None:
                        row.update({
                            "vehicle_type": split_result["vehicle_type"],
                            "route_time": split_result["route_time"],
                            "route_cost": split_result["route_cost"],
                            "route_distance": split_result["route_distance"],
                            "feasible": split_result["feasible"]
                        })
                    else:
                        row.update({
                            "vehicle_type": None,
                            "route_time": np.nan,
                            "route_cost": np.nan,
                            "route_distance": np.nan,
                            "feasible": False
                        })

                    route_rows.append(row)

    return pd.DataFrame(route_rows), current_assign

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

def get_output_by_k(all_outputs, k):
    for item in all_outputs:
        if item["summary"]["k"] == k:
            return item
    return None

def export_plot_data(export_dir, df, result_df, all_outputs, scenarios):
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    result_export = result_df.copy()
    result_export["facility_ids"] = result_export["facility_ids"].apply(
        lambda x: ",".join(map(str, x)) if isinstance(x, (list, tuple)) else x
    )
    result_export.to_csv(export_dir / "result_df.csv", index=False, encoding="utf-8-sig")

    for k in (4, 5):
        output = get_output_by_k(all_outputs, k)
        if output is None:
            continue

        output["assign_df"].to_csv(export_dir / f"k{k}_assign_df.csv", index=False, encoding="utf-8-sig")
        output["equipment_df"].to_csv(export_dir / f"k{k}_equipment_df.csv", index=False, encoding="utf-8-sig")
        output["scenario_df"].to_csv(export_dir / f"k{k}_scenario_df.csv", index=False, encoding="utf-8-sig")
        output["fixed_df"].to_csv(export_dir / f"k{k}_fixed_df.csv", index=False, encoding="utf-8-sig")

    k5_output = get_output_by_k(all_outputs, 5)
    if k5_output is not None:
        with (export_dir / "k5_summary.json").open("w", encoding="utf-8") as f:
            json.dump(make_json_safe(k5_output["summary"]), f, ensure_ascii=False, indent=2)

        route_df, representative_assign_df = build_representative_route_df(
            df,
            k5_output["assign_df"],
            k5_output["equipment_df"],
            k5_output["summary"]["facility_ids"],
            scenarios,
            month=12
        )
        route_df.to_csv(export_dir / "k5_route_df.csv", index=False, encoding="utf-8-sig")
        representative_assign_df.to_csv(export_dir / "k5_assign_representative_df.csv", index=False, encoding="utf-8-sig")

# =========================================================
# 14. 主流程
# =========================================================
def run_full_optimization(df, n_scenarios=300, k_list=(2, 3, 4, 5), seed=42):
    scenarios = generate_scenarios(df, n_scenarios=n_scenarios, seed=seed)
    schemes = generate_candidate_facility_schemes(df, k_list=k_list, random_state=seed)

    all_outputs = []

    for scheme in schemes:
        summary, fixed_df, scenario_df, assign_df, equipment_df = simulate_scheme(df, scheme, scenarios)
        summary["robust_score"] = robust_score(summary)

        all_outputs.append({
            "summary": summary,
            "fixed_df": fixed_df,
            "scenario_df": scenario_df,
            "assign_df": assign_df,
            "equipment_df": equipment_df
        })

    result_df = pd.DataFrame([x["summary"] for x in all_outputs])
    result_df = result_df.sort_values("robust_score").reset_index(drop=True)

    best_summary = result_df.iloc[0].to_dict()
    best_output = None
    for item in all_outputs:
        if item["summary"]["k"] == best_summary["k"] and item["summary"]["facility_ids"] == best_summary["facility_ids"]:
            best_output = item
            break

    return result_df, best_output, all_outputs, scenarios

# =========================================================
# 15. 主程序
# =========================================================
if __name__ == "__main__":
    result_df, best_output, all_outputs, scenarios = run_full_optimization(
        df_stores,
        n_scenarios=300,
        k_list=(2, 3, 4, 5),
        seed=42
    )

    export_dir = Path(__file__).resolve().parent / "导出_升级三"
    export_plot_data(export_dir, df_stores, result_df, all_outputs, scenarios)

    print("\n===================== 第三轮升级版：鲁棒方案对比结果 =====================")
    print(result_df)

    print("\n===================== 第三轮升级版：最优方案摘要 =====================")
    print(best_output["summary"])

    print("\n===================== 第三轮升级版：最优方案设备配置 =====================")
    print(best_output["equipment_df"])

    print("\n===================== 第三轮升级版：最优方案固定成本 =====================")
    print(best_output["fixed_df"])

    print("\n===================== 第三轮升级版：最优方案门店分配 =====================")
    print(best_output["assign_df"])

    print("\n===================== 第三轮升级版：最优方案场景结果（前10行） =====================")
    print(best_output["scenario_df"].head(10))

    print("\nEXPORT_DIR:")
    print(str(export_dir))
