# """
# composite_i_girder_design.py

# 用途：
#   参数化生成式设计 — 钢-混组合工字型主梁（简支单跨示例）
#   - 决策变量: h, bf, tf, tw, t_slab, n_girder (通过间距间接)
#   - 目标: 最小化材料成本（钢 + 混凝土）并满足强度/挠度/构造约束
#   - 优化: 使用差分进化（global optimizer）
#   - 输出: 最优截面参数、关键验算值（M_max, δ_max, 估算钢量/混凝土量, 成本）

# 注意：
#   - 这是工程级快速估算工具用于设计探索（概念设计 / 参数筛选）。
#   - 最终工程设计必须按当地规范（荷载细化、影响线、局部屈曲、疲劳、焊接、连接等）做详细验算并由资质工程师复核。
# """

# import numpy as np
# from scipy.optimize import differential_evolution

# # ---------------------------
# # 常量与默认参数（可修改）
# # ---------------------------
# rho_steel = 7850.0          # kg/m^3
# steel_cost_per_kg = 6.0    # currency/kg (示例)
# conc_cost_per_m3 = 500.0   # currency/m^3 (示例)

# Es = 210e3                 # MPa
# Ec = 30e3                  # MPa (approx for C30)
# n_trans = Es / Ec          # 混凝土转换系数 (to steel equivalent)

# # 车辆/规范荷载（示例简化，按匀布考虑）
# q_dead_per_m = 6.0   # kN/m per girder (structure selfweight + slab etc approximate)
# q_live_per_m = 10.0  # kN/m per girder (规范活载折算到每根梁上的等效值，示例)

# impact_factor = 1.2  # 活载冲击系数

# # Limits / constructability
# MAX_HEIGHT = 2500.0  # mm
# MIN_HEIGHT = 400.0   # mm

# MAX_WING = 800.0     # mm
# MIN_WING = 100.0     # mm

# MIN_THICK = 6.0      # mm
# MAX_THICK = 60.0     # mm

# # 服务性挠度限制比
# DEFLECTION_LIMIT_RATIO = 500.0  # L / 500

# # 其它参数
# g = 9.81  # m/s^2

# # ---------------------------
# # 基本几何与荷载输入（示例）
# # ---------------------------
# class DesignCase:
#     def __init__(self,
#                  L=30.0,          # 跨径 m
#                  B_net=12.0,      # 净宽 m
#                  n_girder=4,      # 主梁根数
#                  slab_thickness_init=0.16  # m
#                  ):
#         self.L = L
#         self.B_net = B_net
#         self.n_girder = n_girder
#         self.beam_spacing = B_net / (n_girder - 1)  # m
#         self.slab_thickness_init = slab_thickness_init

# # ---------------------------
# # 几何与截面计算函数（米与毫米需注意单位统一）
# # ---------------------------
# def section_props_I_steel(h_mm, bf_mm, tf_mm, tw_mm):
#     """
#     计算单个工字钢截面的几何性质（以 mm 单位输入）
#     简化：两个矩形翼缘 + 一个矩形腹板，未考虑倒角与加强筋
#     返回：area (m^2), Ixx (m^4), S_top (m^3)
#     """
#     # convert to meters
#     h = h_mm / 1000.0
#     bf = bf_mm / 1000.0
#     tf = tf_mm / 1000.0
#     tw = tw_mm / 1000.0

#     # wings (top and bottom)
#     A_wing = bf * tf
#     # web (腹板) 高度 = 总高 - 2*翼缘厚
#     hw = max(h - 2 * tf, 0.0)
#     A_web = hw * tw

#     A_total = 2 * A_wing + A_web  # m^2

#     # centroid at mid-height -> symmetric -> neutral axis at mid-height
#     # I for rectangle about its centroid: (b*h^3)/12
#     I_wing = (bf * tf**3) / 12.0
#     # distance from wing centroid to mid axis
#     y_wing = (h / 2.0 - tf / 2.0)
#     I_wing_total = 2 * (I_wing + A_wing * y_wing**2)

#     I_web = (tw * hw**3) / 12.0  # about web centroid at mid
#     # web centroid is at mid -> no parallel axis term
#     I_total = I_wing_total + I_web

#     # section modulus S = I / (h/2)
#     S = I_total / (h / 2.0) if h > 0 else 1e-9

#     return A_total, I_total, S

# def transformed_concrete_slab_I(slab_t_m, width_per_beam_m):
#     """
#     计算混凝土上板（按每根梁分摊宽度）的等效钢惯性矩（转换法）
#     slab_t_m: 混凝土厚度 m
#     width_per_beam_m: 分摊到每根梁的板宽 m
#     返回: A_conc (m^2), I_conc_transformed (m^4), centroid位置相对于钢截面中性轴（已假定钢中性轴在截面中间）
#     注：此处为简化，将混凝土板置于钢翼缘上方（构造上常用），中性轴位置需组合计算，脚本使用简化方法：
#       - 假定钢截面中性轴在工字截面中间高度（对称）；
#       - 混凝土板顶面距中性轴 = h/2 + slab_t/2
#     这对初步设计是可接受的估算；详细计算应求组合截面中性轴位置再计算 I。
#     """
#     A_conc = slab_t_m * width_per_beam_m
#     # assume slab centroid distance to steel mid axis:
#     # Note: user of function should ensure consistent assumptions
#     # We'll return I transformed approx: I = A * d^2 + I_rect/12
#     I_rect = (width_per_beam_m * slab_t_m**3) / 12.0
#     # The distance d between slab centroid and steel mid-height will be provided by caller as needed.
#     return A_conc, I_rect

# # ---------------------------
# # 评估函数：给定设计变量, 返回目标与约束违例（用于优化）
# # ---------------------------
# def evaluate_design(x, case: DesignCase, verbose=False):
#     """
#     x: 决策向量
#        x[0] = h_mm
#        x[1] = bf_mm
#        x[2] = tf_mm
#        x[3] = tw_mm
#        x[4] = slab_t_mm
#        x[5] = beam_spacing_m_factor (相对于 case.beam_spacing 的比例，用来允许改变梁间距)
#     """
#     # 解码变量
#     h_mm = x[0]
#     bf_mm = x[1]
#     tf_mm = x[2]
#     tw_mm = x[3]
#     slab_t_mm = x[4]
#     spacing_factor = x[5]

#     # enforce constructability via simple bounds (在优化边界以外的仍会被惩罚)
#     h_mm = np.clip(h_mm, MIN_HEIGHT, MAX_HEIGHT)
#     bf_mm = np.clip(bf_mm, MIN_WING, MAX_WING)
#     tf_mm = np.clip(tf_mm, MIN_THICK, MAX_THICK)
#     tw_mm = np.clip(tw_mm, MIN_THICK, MAX_THICK)
#     slab_t_mm = max(slab_t_mm, 50.0)  # 最薄 50 mm

#     beam_spacing = case.beam_spacing * spacing_factor
#     # ensure spacing reasonable (not exceed bridge width)
#     beam_spacing = np.clip(beam_spacing, 2.0, case.B_net)  # meters

#     # 单根梁分摊板宽（简化：等分板宽）
#     width_per_beam = beam_spacing

#     # 截面性质（钢）
#     A_s, I_s, S_s = section_props_I_steel(h_mm, bf_mm, tf_mm, tw_mm)  # area m^2, I m^4, S m^3

#     # 混凝土板（转换为等效钢）
#     slab_t_m = slab_t_mm / 1000.0
#     A_conc, I_rect = transformed_concrete_slab_I(slab_t_m, width_per_beam)

#     # 转换混凝土面积到等效钢面积
#     A_conc_eq = n_trans * A_conc
#     # 假定混凝土板质心距钢中性轴 d (approx)：
#     h_m = h_mm / 1000.0
#     # 混凝土板位于钢截面上方，质心距 steel mid = h/2 + slab_t/2
#     d = h_m / 2.0 + slab_t_m / 2.0
#     # 平行轴定理: I_transformed = n*(I_rect + A_conc * d^2)
#     I_conc_trans = n_trans * (I_rect + A_conc * d**2)

#     # 等效组合截面惯性矩（转换到钢）
#     I_eq = I_s + I_conc_trans
#     # 等效截面模数（以全高度 h）
#     S_eq = I_eq / (h_m / 2.0)

#     # -----------------------
#     # 荷载计算（简化：均布载）
#     # total distributed load per unit length per beam (kN/m)
#     q_total = q_dead_per_m + impact_factor * q_live_per_m  # kN/m per beam
#     # Convert to N/m for use with SI E (Pa) and I (m^4): 1 kN = 1000 N
#     q_N_per_m = q_total * 1000.0

#     # 跨中最大弯矩（简支匀布）
#     L = case.L
#     M_max = q_N_per_m * L**2 / 8.0  # N*m

#     # 弯曲应力（等效钢）， sigma = M / S
#     sigma_max = M_max / S_eq  # N/m^2 = Pa
#     sigma_max_MPa = sigma_max / 1e6

#     # 估算最大挠度（弹性简支匀布）： delta = 5 q L^4 / (384 E I)
#     delta_max = (5.0 * q_N_per_m * L**4) / (384.0 * Es * 1e6 * I_eq)  # Es in MPa -> Pa = MPa*1e6
#     # delta_max in meters

#     # -----------------------
#     # 强度与挠度约束（示例简单处理）
#     fy = 355.0  # MPa (S355)
#     # 强度安全裕度: sigma_max <= 0.9*fy (示例)
#     strength_ok = sigma_max_MPa <= 0.9 * fy

#     # 挠度约束
#     delta_allow = L / DEFLECTION_LIMIT_RATIO  # meters
#     deflection_ok = delta_max <= delta_allow

#     # 局部构造约束（跨/高比软约束）
#     span_to_depth = L / h_m
#     span_depth_ok = (span_to_depth >= 10.0) and (span_to_depth <= 25.0)  # 宽松范围

#     # -----------------------
#     # 材料量与成本估计
#     steel_volume = A_s * L  # m^3 per beam
#     steel_mass = steel_volume * rho_steel  # kg per beam
#     total_steel_mass = steel_mass * case.n_girder

#     conc_volume_per_beam = A_conc * L  # m^3 per beam (板体积作为主要混凝土用量)
#     total_conc_volume = conc_volume_per_beam * case.n_girder

#     cost = total_steel_mass * steel_cost_per_kg + total_conc_volume * conc_cost_per_m3

#     # 违约罚项（若违反约束则大幅罚分）
#     penalty = 0.0
#     if not strength_ok:
#         # 超出比例
#         violation_ratio = (sigma_max_MPa / (0.9 * fy))
#         penalty += 1e6 * (violation_ratio - 1.0) if violation_ratio > 1.0 else 0.0
#     if not deflection_ok:
#         violation_ratio = (delta_max / delta_allow)
#         penalty += 5e5 * (violation_ratio - 1.0) if violation_ratio > 1.0 else 0.0
#     if not span_depth_ok:
#         # 若不在合理范围，加小罚分（鼓励满足）
#         penalty += 1e4

#     # 目标函数值 = 成本 + 罚项
#     objective = cost + penalty

#     if verbose:
#         print("Design vars (mm): h,bf,tf,tw,slab_t =", h_mm, bf_mm, tf_mm, tw_mm, slab_t_mm)
#         print("beam spacing (m):", beam_spacing)
#         print("A_s (m2), I_s (m4), S_s (m3):", A_s, I_s, S_s)
#         print("I_eq (m4):", I_eq)
#         print("M_max (kN*m):", M_max / 1e3)
#         print("sigma_max (MPa):", sigma_max_MPa)
#         print("delta_max (mm):", delta_max * 1000.0, "allow (mm):", delta_allow * 1000.0)
#         print("steel mass (kg):", total_steel_mass, "conc vol (m3):", total_conc_volume)
#         print("cost:", cost, "penalty:", penalty, "objective:", objective)
#         print("span/depth:", span_to_depth, "OK strength,defl,span:", strength_ok, deflection_ok, span_depth_ok)

#     # 返回 objective 和诊断数据
#     diagnostics = {
#         'objective': objective,
#         'cost': cost,
#         'penalty': penalty,
#         'sigma_max_MPa': sigma_max_MPa,
#         'delta_max_m': delta_max,
#         'delta_allow_m': delta_allow,
#         'total_steel_mass_kg': total_steel_mass,
#         'total_conc_volume_m3': total_conc_volume,
#         'I_eq': I_eq,
#         'S_eq': S_eq,
#         'span_to_depth': span_to_depth,
#         'strength_ok': strength_ok,
#         'deflection_ok': deflection_ok,
#     }
#     return objective, diagnostics

# # ---------------------------
# # 优化器包装
# # ---------------------------
# def optimize_design(case: DesignCase, maxiter=60, popsize=15, verbose=False):
#     # 决策变量边界（用于 differential_evolution）
#     # h_mm: [MIN_HEIGHT, MAX_HEIGHT]
#     # bf_mm: [MIN_WING, MAX_WING]
#     # tf_mm: [MIN_THICK, 60]
#     # tw_mm: [MIN_THICK, 40]
#     # slab_t_mm: [50, 300]  (50mm ~ 300mm)
#     # spacing_factor: [0.7, 1.5] 允许调整梁间距比例
#     bounds = [
#         (MIN_HEIGHT, MAX_HEIGHT),
#         (MIN_WING, MAX_WING),
#         (MIN_THICK, 60.0),
#         (MIN_THICK, 40.0),
#         (50.0, 300.0),
#         (0.8, 1.2),
#     ]

#     def obj(x):
#         val, _ = evaluate_design(x, case, verbose=False)
#         return val

#     result = differential_evolution(obj, bounds, maxiter=maxiter, popsize=popsize, disp=verbose, polish=True, tol=1e-3)
#     x_opt = result.x
#     obj_val, diag = evaluate_design(x_opt, case, verbose=False)
#     return x_opt, obj_val, diag, result

# # ---------------------------
# # 主程序：示例运行
# # ---------------------------
# if __name__ == "__main__":
#     # 示例设计工况
#     case = DesignCase(L=30.0, B_net=12.0, n_girder=4, slab_thickness_init=0.16)

#     print("=== Composite I-Girder automatic design (示例) ===")
#     print("Span L = {:.1f} m, Bridge net width B = {:.1f} m, n_girder = {}".format(case.L, case.B_net, case.n_girder))
#     print("Starting optimization... (this may take a minute depending on maxiter/popsize)")

#     x_opt, obj_val, diag, res = optimize_design(case, maxiter=60, popsize=12, verbose=True)

#     print("\n=== Optimization result ===")
#     print("Optimal decision vector (approx):")
#     print("h_mm = {:.1f}, bf_mm = {:.1f}, tf_mm = {:.1f}, tw_mm = {:.1f}, slab_t_mm = {:.1f}, spacing_factor = {:.3f}".format(
#         x_opt[0], x_opt[1], x_opt[2], x_opt[3], x_opt[4], x_opt[5]
#     ))

#     print("\nKey diagnostics:")
#     print("Objective (cost+penalty): {:.2f}".format(diag['objective']))
#     print("Estimated material cost: {:.2f}".format(diag['cost']))
#     print("Total steel mass (kg): {:.1f}".format(diag['total_steel_mass_kg']))
#     print("Total concrete volume (m3): {:.3f}".format(diag['total_conc_volume_m3']))
#     print("Max bending stress (MPa): {:.2f}".format(diag['sigma_max_MPa']))
#     print("Max deflection (mm): {:.2f}  Allow (mm): {:.2f}".format(diag['delta_max_m']*1000.0, diag['delta_allow_m']*1000.0))
#     print("Span/depth ratio: {:.2f}".format(diag['span_to_depth']))
#     print("Constraint flags - strength_ok: {}, deflection_ok: {}".format(diag['strength_ok'], diag['deflection_ok']))

#     print("\nNote: 结果为概念/初步设计级别的估算。请用有限元和规范化验算做最终设计。")


import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib

# font printing issues
matplotlib.rcParams['font.sans-serif'] = ['Heiti TC']
matplotlib.rcParams['axes.unicode_minus'] = False     # fix minus sign issue

# -------------------------------
# 1. 参数输入（用户输入）
# -------------------------------
def get_bridge_inputs():
    try:
        L = float(input("请输入目标跨径 L (m): "))
        B = float(input("请输入桥面宽度 B (m): "))
    except ValueError:
        print("❌ 输入无效，请输入数值。")
        return None, None
    return L, B


# -------------------------------
# 2. 工字型钢-混组合梁的截面定义
# -------------------------------
def section_properties(h, bf, tf, tw):
    """计算工字梁的截面特性 (近似)"""
    # 截面面积
    A = 2 * bf * tf + (h - 2 * tf) * tw
    # 惯性矩 I（关于中性轴）
    I = (bf * h**3 / 12 - (bf - tw) * (h - 2 * tf)**3 / 12)
    return A, I


# -------------------------------
# 3. 梁桥性能计算函数
# -------------------------------
def bridge_performance(x, L, B, q=30):
    """
    输入: x = [h, bf, tf, tw]
    输出: 挠度, 应力, 刚度, 造价指标
    """
    h, bf, tf, tw = x
    E = 2.0e11  # 钢弹性模量 Pa
    A, I = section_properties(h, bf, tf, tw)
    
    # 均布荷载下简支梁最大挠度 (w = 5qL^4 / (384EI))
    w_max = 5 * q * (L**4) / (384 * E * I)
    # 最大弯矩 Mmax = qL^2 / 8
    Mmax = q * (L**2) / 8
    # 最大正应力 σ = M*y/I, y = h/2
    sigma_max = Mmax * (h / 2) / I
    # 造价指标（简化：重量为主）
    cost = A * L * 7.85e3  # kg

    return w_max, sigma_max, cost


# -------------------------------
# 4. 目标函数与约束
# -------------------------------
def objective(x, L, B):
    """目标函数 = 综合经济性 + 惩罚项"""
    w, sigma, cost = bridge_performance(x, L, B)
    penalty = 0

    # 约束惩罚项
    if w > L / 800:  # 挠度限值
        penalty += 1e8 * (w - L / 800)
    if sigma > 250e6:  # 强度限值
        penalty += 1e8 * (sigma - 250e6)

    return cost + penalty


# -------------------------------
# 5. 优化执行
# -------------------------------
def optimize_bridge(L, B):
    # 初始值 [h, bf, tf, tw]
    x0 = [L / 20, B / 10, 0.02, 0.015]
    bounds = [(L/30, L/15), (B/15, B/5), (0.01, 0.05), (0.005, 0.03)]
    
    res = minimize(objective, x0, args=(L, B), bounds=bounds)
    return res


# -------------------------------
# 6. 绘图函数（示意工字梁截面）
# -------------------------------
def plot_section(h, bf, tf, tw):
    fig, ax = plt.subplots()
    ax.plot([-bf/2, bf/2], [h/2, h/2], 'k-', lw=3)      # 上翼缘
    ax.plot([-tw/2, tw/2], [-h/2, h/2], 'k-', lw=3)     # 腹板
    ax.plot([-bf/2, bf/2], [-h/2, -h/2], 'k-', lw=3)    # 下翼缘
    ax.set_aspect('equal')
    ax.set_xlabel('宽度 (m)')
    ax.set_ylabel('高度 (m)')
    ax.set_title('最优工字型钢-混组合梁截面')
    plt.grid(True)
    plt.show()


# -------------------------------
# 7. 主程序
# -------------------------------
if __name__ == "__main__":
    L, B = get_bridge_inputs()
    if L and B:
        print(f"\n🚧 开始优化设计：L={L:.2f} m, B={B:.2f} m ...")
        res = optimize_bridge(L, B)
        h, bf, tf, tw = res.x
        w, sigma, cost = bridge_performance(res.x, L, B)

        print("\n✅ 最优设计结果：")
        print(f"主梁高度 h = {h:.3f} m")
        print(f"翼缘宽度 bf = {bf:.3f} m")
        print(f"翼缘厚度 tf = {tf:.3f} m")
        print(f"腹板厚度 tw = {tw:.3f} m")
        print(f"最大挠度 = {w*1000:.3f} mm (限值 {L/800:.3f} m)")
        print(f"最大应力 = {sigma/1e6:.2f} MPa (限值 250 MPa)")
        print(f"造价指标 ≈ {cost/1e3:.2f} 吨钢\n")

        plot_section(h, bf, tf, tw)
