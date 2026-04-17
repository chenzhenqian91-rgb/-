import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2, ceil
from sklearn.cluster import KMeans
import random

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
SERVICE_LEVEL = 0.95
SAFETY_MARGIN = 0.10
MAX_ROUTE_HOURS = 3.0

LATE_PENALTY = 5000
SHORTAGE_PENALTY = 8000

VEHICLES = {
    "sanlun": {
        "capacity": 500,
        "speed_kmh": 20,
        "cost_func": lambda stops: 170 + 30 * stops
    },
    "jinbei": {
        "capacity": 1000,
        "speed_kmh": 30,
        "cost_func": lambda stops: 280 + 30 * stops
    },
    "rider": {
        "capacity": 50,
        "speed_kmh": 15,
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
    elif equipment_num == 1:
        return 50
    return 50 + (equipment_num - 1) * 40

def route_distance_with_return(proc_lon, proc_lat, route_stores):
    if len(route_stores) == 0:
        return 0.0

    total = 0.0
    prev_lon, prev_lat = proc_lon, proc_lat

    for _, row in route_stores.iterrows():
        total += haversine_km(prev_lon, prev_lat, row["longitude"], row["latitude"])
        prev_lon, prev_lat = row["longitude"], row["latitude"]

    total += haversine_km(prev_lon, prev_lat, proc_lon, proc_lat)
    return total

def polar_angle(cx, cy, x, y):
    return np.arctan2(y - cy, x - cx)

# =========================================================
# 4. 多周期需求与场景生成
# =========================================================
def generate_base_monthly_demand():
    base = {}
    for t in MONTHS:
        base[t] = BASE_DEMAND_START + (BASE_DEMAND_END - BASE_DEMAND_START) * (t - 1) / 11
    return base

def generate_store_factors(df, seed=42):
    rng = np.random.default_rng(seed)
    factors = rng.uniform(0.85, 1.15, size=len(df))
    return dict(zip(df["store_id"], factors))

def generate_scenarios(df, n_scenarios=100, seed=42):
    rng = np.random.default_rng(seed)
    base_monthly = generate_base_monthly_demand()
    store_factors = generate_store_factors(df, seed=seed)

    scenarios = []

    for s in range(n_scenarios):
        demand_dict = {}
        travel_factor = {}
        service_time = {}

        for month in MONTHS:
            month_std = 0.08 if month <= 8 else 0.15
            for store_id in df["store_id"]:
                xi = np.clip(rng.normal(1.0, month_std), 0.75, 1.35)
                demand = base_monthly[month] * store_factors[store_id] * xi
                demand_dict[(store_id, month)] = max(1, demand)

        for i in df["store_id"]:
            service_time[i] = rng.uniform(4, 10) / 60.0  # 小时

        for i in df["store_id"]:
            for j in df["store_id"]:
                if i != j:
                    tf = rng.choice([1.0, 1.15, 1.30, 1.50], p=[0.55, 0.25, 0.15, 0.05])
                    travel_factor[(i, j)] = tf

        scenarios.append({
            "scenario_id": s,
            "demand": demand_dict,
            "travel_factor": travel_factor,
            "service_time": service_time
        })

    return scenarios

# =========================================================
# 5. 候选选址方案生成（聚类仅作初始方案）
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

        schemes.append({
            "k": k,
            "facility_ids": facilities
        })

    return schemes

# =========================================================
# 6. 门店分配：主服务 + 备选服务
# =========================================================
def assign_stores_with_backup(df, facility_ids):
    facility_df = df[df["store_id"].isin(facility_ids)].copy()

    assignments = []
    for _, row in df.iterrows():
        sid = row["store_id"]

        distances = []
        for _, frow in facility_df.iterrows():
            fid = frow["store_id"]
            dist = haversine_km(row["longitude"], row["latitude"], frow["longitude"], frow["latitude"])
            distances.append((fid, dist))

        distances = sorted(distances, key=lambda x: x[1])
        primary = distances[0][0]
        backup = distances[1][0] if len(distances) > 1 else primary

        assignments.append({
            "store_id": sid,
            "primary_facility": primary,
            "backup_facility": backup,
            "dist_primary": distances[0][1],
            "dist_backup": distances[1][1] if len(distances) > 1 else distances[0][1]
        })

    return pd.DataFrame(assignments)

# =========================================================
# 7. 设备配置（按高分位需求 + 安全冗余）
# =========================================================
def configure_equipment(df_assign, scenarios, facility_ids):
    equipment_rows = []

    for fid in facility_ids:
        served_stores = df_assign[df_assign["primary_facility"] == fid]["store_id"].tolist()
        monthly_loads = []

        for scenario in scenarios:
            for month in MONTHS:
                total_demand = sum(scenario["demand"][(sid, month)] for sid in served_stores)
                monthly_loads.append(total_demand)

        robust_load = np.quantile(monthly_loads, SERVICE_LEVEL) * (1 + SAFETY_MARGIN)
        equipment_num = ceil(robust_load / CAPACITY_PER_EQUIPMENT)

        equipment_rows.append({
            "facility_id": fid,
            "robust_load": robust_load,
            "equipment_num": equipment_num
        })

    return pd.DataFrame(equipment_rows)

# =========================================================
# 8. 边界门店重分配
# =========================================================
def reassign_boundary_stores(df_assign, threshold_km=8):
    df_assign = df_assign.copy()
    df_assign["is_boundary"] = (df_assign["dist_backup"] - df_assign["dist_primary"]) <= threshold_km
    return df_assign

# =========================================================
# 9. 路径构造：极角排序 + 容量拆分
# =========================================================
def build_routes_for_facility(df, facility_id, assigned_store_ids, demand_per_store, vehicle_capacity):
    facility_row = df[df["store_id"] == facility_id].iloc[0]
    proc_lon, proc_lat = facility_row["longitude"], facility_row["latitude"]

    target_df = df[df["store_id"].isin(assigned_store_ids) & (df["store_id"] != facility_id)].copy()
    if target_df.empty:
        return []

    target_df["angle"] = target_df.apply(
        lambda r: polar_angle(proc_lon, proc_lat, r["longitude"], r["latitude"]),
        axis=1
    )
    target_df = target_df.sort_values("angle").reset_index(drop=True)

    routes = []
    current = []
    current_load = 0

    for _, row in target_df.iterrows():
        if current_load + demand_per_store[row["store_id"]] <= vehicle_capacity:
            current.append(row["store_id"])
            current_load += demand_per_store[row["store_id"]]
        else:
            routes.append(current)
            current = [row["store_id"]]
            current_load = demand_per_store[row["store_id"]]

    if current:
        routes.append(current)

    return routes

# =========================================================
# 10. 2-opt 局部优化
# =========================================================
def compute_route_distance_by_ids(df, facility_id, route_ids):
    facility_row = df[df["store_id"] == facility_id].iloc[0]
    proc_lon, proc_lat = facility_row["longitude"], facility_row["latitude"]
    route_df = df[df["store_id"].isin(route_ids)].copy()

    ordered = []
    for rid in route_ids:
        ordered.append(df[df["store_id"] == rid].iloc[0])

    total = 0.0
    prev_lon, prev_lat = proc_lon, proc_lat
    for row in ordered:
        total += haversine_km(prev_lon, prev_lat, row["longitude"], row["latitude"])
        prev_lon, prev_lat = row["longitude"], row["latitude"]
    total += haversine_km(prev_lon, prev_lat, proc_lon, proc_lat)
    return total

def two_opt_route(df, facility_id, route_ids):
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
# 11. 路线车型选择与时间计算（含返程）
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
            total_distance += leg_dist
            tf = scenario["travel_factor"].get((prev_id, sid), 1.0)
            total_time += leg_dist / info["speed_kmh"] * tf
            total_time += scenario["service_time"][sid]

            prev_id = sid
            prev_lon, prev_lat = row["longitude"], row["latitude"]

        # 返程
        back_dist = haversine_km(prev_lon, prev_lat, proc_lon, proc_lat)
        total_distance += back_dist
        tf_back = scenario["travel_factor"].get((prev_id, facility_id), 1.0)
        total_time += back_dist / info["speed_kmh"] * tf_back

        if total_time > MAX_ROUTE_HOURS:
            continue

        if vehicle_name == "rider":
            route_cost = total_time * info["hourly_cost"]
        else:
            route_cost = info["cost_func"](num_stops)

        option = {
            "vehicle_type": vehicle_name,
            "route_cost": route_cost,
            "route_time": total_time,
            "route_distance": total_distance
        }

        if (best_option is None) or (route_cost < best_option["route_cost"]):
            best_option = option

    return best_option

# =========================================================
# 12. 构造并评估全方案
# =========================================================
def simulate_scheme(df, scheme, scenarios, equipment_life_years=3):
    facility_ids = scheme["facility_ids"]
    df_assign = assign_stores_with_backup(df, facility_ids)
    df_assign = reassign_boundary_stores(df_assign, threshold_km=8)
    equipment_df = configure_equipment(df_assign, scenarios, facility_ids)

    # 固定成本
    fixed_rows = []
    for _, row in equipment_df.iterrows():
        fid = row["facility_id"]
        eq_num = int(row["equipment_num"])
        annual_rent = calc_daily_rent(eq_num) * DAYS_PER_YEAR
        annual_equipment = eq_num * EQUIPMENT_COST / equipment_life_years
        fixed_rows.append({
            "facility_id": fid,
            "equipment_num": eq_num,
            "annual_rent": annual_rent,
            "annual_equipment_cost": annual_equipment,
            "annual_fixed_cost": annual_rent + annual_equipment
        })

    fixed_df = pd.DataFrame(fixed_rows)
    total_fixed_cost = fixed_df["annual_fixed_cost"].sum()

    scenario_results = []

    for scenario in scenarios:
        transport_cost = 0.0
        late_count = 0
        shortage_count = 0
        total_routes = 0

        # 只取12月高峰做每日运行仿真，也可扩展到逐月平均
        demand_per_store = {sid: scenario["demand"][(sid, 12)] for sid in df["store_id"]}

        # 如果主加工店超载，边界门店切换到备选加工店
        current_assign = df_assign.copy()

        # 简单负载检查
        for fid in facility_ids:
            served = current_assign[current_assign["primary_facility"] == fid]["store_id"].tolist()
            load = sum(demand_per_store[sid] for sid in served)
            eq_num = int(equipment_df[equipment_df["facility_id"] == fid]["equipment_num"].iloc[0])
            cap = eq_num * CAPACITY_PER_EQUIPMENT

            if load > cap:
                boundary = current_assign[
                    (current_assign["primary_facility"] == fid) & (current_assign["is_boundary"])
                ].copy()

                boundary = boundary.sort_values("dist_backup" - "dist_primary" if False else "dist_backup")

                for idx, brow in boundary.iterrows():
                    sid = brow["store_id"]
                    backup = brow["backup_facility"]
                    backup_eq = int(equipment_df[equipment_df["facility_id"] == backup]["equipment_num"].iloc[0])

                    served_backup = current_assign[current_assign["primary_facility"] == backup]["store_id"].tolist()
                    backup_load = sum(demand_per_store[x] for x in served_backup)
                    if backup_load + demand_per_store[sid] <= backup_eq * CAPACITY_PER_EQUIPMENT:
                        current_assign.loc[current_assign["store_id"] == sid, "primary_facility"] = backup
                        load -= demand_per_store[sid]
                    if load <= cap:
                        break

        # 生成路径
        for fid in facility_ids:
            served = current_assign[current_assign["primary_facility"] == fid]["store_id"].tolist()
            if fid not in served:
                served = served + [fid]

            # 先以三轮容量拆分，后面车型再选
            routes = build_routes_for_facility(
                df, fid, served, demand_per_store, vehicle_capacity=500
            )

            for route in routes:
                if len(route) == 0:
                    continue

                # 2-opt优化
                if len(route) >= 3:
                    route = two_opt_route(df, fid, route)

                result = evaluate_route(df, route, fid, scenario, demand_per_store)

                # 若三轮拆线后仍不可行，尝试单点拆分
                if result is None:
                    repaired = True
                    for sid in route:
                        single_result = evaluate_route(df, [sid], fid, scenario, demand_per_store)
                        total_routes += 1
                        if single_result is None:
                            late_count += 1
                            shortage_count += 1
                            transport_cost += LATE_PENALTY + SHORTAGE_PENALTY
                            repaired = False
                        else:
                            transport_cost += single_result["route_cost"]
                            if single_result["route_time"] > MAX_ROUTE_HOURS:
                                late_count += 1
                                transport_cost += LATE_PENALTY
                    continue

                total_routes += 1
                transport_cost += result["route_cost"]
                if result["route_time"] > MAX_ROUTE_HOURS:
                    late_count += 1
                    transport_cost += LATE_PENALTY

        annual_transport_cost = transport_cost * DAYS_PER_YEAR

        scenario_results.append({
            "scenario_id": scenario["scenario_id"],
            "annual_transport_cost": annual_transport_cost,
            "annual_total_cost": annual_transport_cost + total_fixed_cost,
            "late_count": late_count,
            "shortage_count": shortage_count,
            "total_routes": total_routes
        })

    scenario_df = pd.DataFrame(scenario_results)

    mean_total_cost = scenario_df["annual_total_cost"].mean()
    p95_total_cost = scenario_df["annual_total_cost"].quantile(0.95)
    worst_total_cost = scenario_df["annual_total_cost"].max()
    late_rate = (scenario_df["late_count"] > 0).mean()
    shortage_rate = (scenario_df["shortage_count"] > 0).mean()

    # 简单CVaR
    threshold = scenario_df["annual_total_cost"].quantile(0.95)
    cvar = scenario_df[scenario_df["annual_total_cost"] >= threshold]["annual_total_cost"].mean()

    summary = {
        "k": scheme["k"],
        "facility_ids": facility_ids,
        "total_fixed_cost": total_fixed_cost,
        "mean_total_cost": mean_total_cost,
        "p95_total_cost": p95_total_cost,
        "worst_total_cost": worst_total_cost,
        "cvar95": cvar,
        "late_rate": late_rate,
        "shortage_rate": shortage_rate
    }

    return summary, fixed_df, scenario_df, df_assign, equipment_df

# =========================================================
# 13. 鲁棒决策
# =========================================================
def robust_score(summary, w_mean=1.0, w_cvar=0.3, w_late=50000, w_shortage=80000):
    return (
        w_mean * summary["mean_total_cost"]
        + w_cvar * summary["cvar95"]
        + w_late * summary["late_rate"]
        + w_shortage * summary["shortage_rate"]
    )

def run_full_optimization(df, n_scenarios=100, k_list=(2, 3, 4, 5), equipment_life_years=3, seed=42):
    scenarios = generate_scenarios(df, n_scenarios=n_scenarios, seed=seed)
    schemes = generate_candidate_facility_schemes(df, k_list=k_list, random_state=seed)

    all_outputs = []

    for scheme in schemes:
        summary, fixed_df, scenario_df, df_assign, equipment_df = simulate_scheme(
            df, scheme, scenarios, equipment_life_years=equipment_life_years
        )
        summary["robust_score"] = robust_score(summary)
        all_outputs.append({
            "summary": summary,
            "fixed_df": fixed_df,
            "scenario_df": scenario_df,
            "assign_df": df_assign,
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

    return result_df, best_output, all_outputs

# =========================================================
# 14. 主程序
# =========================================================
if __name__ == "__main__":
    result_df, best_output, all_outputs = run_full_optimization(
        df_stores,
        n_scenarios=200,
        k_list=(2, 3, 4, 5),
        equipment_life_years=3,
        seed=42
    )

    print("\n===================== 鲁棒方案对比结果 =====================")
    print(result_df)

    print("\n===================== 最优方案摘要 =====================")
    print(best_output["summary"])

    print("\n===================== 最优方案设备配置 =====================")
    print(best_output["equipment_df"])

    print("\n===================== 最优方案固定成本 =====================")
    print(best_output["fixed_df"])

    print("\n===================== 最优方案门店分配 =====================")
    print(best_output["assign_df"].head(30))

    print("\n===================== 最优方案场景结果（前10行） =====================")
    print(best_output["scenario_df"].head(10))