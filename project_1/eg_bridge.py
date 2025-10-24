import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import random

# 选择某一种桥型，采用生成式设计方法，产生桥梁的初步结构方案

# which bridge is one enough or i gotta have like bridge 1 2 3 for different scenarios
# do i have to consider other factors eg weight wind sea
# need to consider cost?

class BridgeDesign:
    """桥梁设计个体"""
    def __init__(self, params=None):
        # 设计参数: [主梁高度(m), 桥墩间距(m), 桥墩数量, 材料类型]
        if params is None:
            self.params = self.random_params()
        else:
            self.params = params
        self.fitness = 0
        self.weight = 0
        self.stress = 0
        self.deflection = 0
    
    def random_params(self):
        """生成随机参数"""
        return [
            np.random.uniform(0.5, 2.0),    # 主梁高度 0.5-2.0m
            np.random.uniform(15, 40),      # 桥墩间距 15-40m
            random.randint(2, 6),           # 桥墩数量 2-6个
            random.randint(0, 2)            # 材料类型 0:混凝土, 1:钢, 2:组合
        ]
    
    def calculate_performance(self, span=100):
        """计算结构性能（简化计算）"""
        beam_height, pier_spacing, num_piers, material_type = self.params
        
        # 总跨度
        total_span = pier_spacing * (num_piers - 1)
        
        # 材料属性
        material_props = {
            0: {'density': 2500, 'elasticity': 3e4, 'strength': 30},  # 混凝土
            1: {'density': 7850, 'elasticity': 2e5, 'strength': 235}, # 钢
            2: {'density': 4500, 'elasticity': 1e5, 'strength': 150}  # 组合
        }
        
        prop = material_props[material_type]
        
        # 简化重量计算 (kg/m)
        unit_weight = beam_height * 3 * prop['density']  # 假设桥宽3m
        
        # 总重量
        self.weight = unit_weight * total_span
        
        # 简化应力计算 (MPa)
        # 假设均布荷载 10 kN/m (自重+活载)
        total_load = 10000 * total_span  # N
        moment = total_load * pier_spacing / 8  # 简支梁弯矩
        section_modulus = 3 * beam_height**2 / 6  # 简化截面模量
        self.stress = moment / section_modulus / 1e6  # MPa
        
        # 简化挠度计算 (mm)
        inertia = 3 * beam_height**3 / 12  # 截面惯性矩
        self.deflection = (5 * 10000 * pier_spacing**4) / (384 * prop['elasticity'] * inertia) * 1000
        
        return self.weight, self.stress, self.deflection

class GenerativeBridgeDesign:
    """生成式桥梁设计优化器"""
    
    def __init__(self, population_size=50, generations=100):
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.best_designs = []
        self.fitness_history = []
        
    def evaluate_fitness(self, design):
        """评估设计适应度"""
        weight, stress, deflection = design.calculate_performance()
        
        # 约束条件
        max_stress = 200  # MPa
        max_deflection = span / 400 * 1000  # mm
        
        # 惩罚项
        stress_penalty = max(0, stress - max_stress) * 1000
        deflection_penalty = max(0, deflection - max_deflection) * 1000
        
        # 适应度函数: 最小化重量，同时满足约束
        fitness = 1 / (weight + stress_penalty + deflection_penalty + 1)
        
        design.fitness = fitness
        return fitness
    
    def initialize_population(self):
        """初始化种群"""
        self.population = [BridgeDesign() for _ in range(self.population_size)]
    
    def selection(self, tournament_size=3):
        """锦标赛选择"""
        selected = []
        for _ in range(self.population_size):
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)
        return selected
    
    def crossover(self, parent1, parent2):
        """交叉操作"""
        child_params = []
        for i in range(len(parent1.params)):
            if random.random() < 0.5:
                child_params.append(parent1.params[i])
            else:
                child_params.append(parent2.params[i])
        return BridgeDesign(child_params)
    
    def mutation(self, design, mutation_rate=0.1):
        """变异操作"""
        mutated_params = design.params.copy()
        for i in range(len(mutated_params)):
            if random.random() < mutation_rate:
                if i == 0:  # 主梁高度
                    mutated_params[i] = np.random.uniform(0.5, 2.0)
                elif i == 1:  # 桥墩间距
                    mutated_params[i] = np.random.uniform(15, 40)
                elif i == 2:  # 桥墩数量
                    mutated_params[i] = random.randint(2, 6)
                elif i == 3:  # 材料类型
                    mutated_params[i] = random.randint(0, 2)
        return BridgeDesign(mutated_params)
    
    def evolve(self):
        """进化过程"""
        self.initialize_population()
        
        for generation in range(self.generations):
            # 评估适应度
            for design in self.population:
                self.evaluate_fitness(design)
            
            # 记录最佳个体
            best_design = max(self.population, key=lambda x: x.fitness)
            self.best_designs.append(best_design)
            avg_fitness = np.mean([d.fitness for d in self.population])
            self.fitness_history.append(avg_fitness)
            
            print(f"Generation {generation}: Best Fitness = {best_design.fitness:.6f}, "
                  f"Avg Fitness = {avg_fitness:.6f}")
            
            # 选择
            selected = self.selection()
            
            # 交叉和变异
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % len(selected)]
                
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.population_size]
    
    def visualize_results(self):
        """可视化结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 适应度进化历史
        axes[0, 0].plot(self.fitness_history)
        axes[0, 0].set_title('Average Fitness Evolution')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Fitness')
        
        # 最佳设计参数分布
        best_params = np.array([d.params for d in self.best_designs])
        axes[0, 1].scatter(best_params[:, 0], best_params[:, 1], c=range(len(best_params)), cmap='viridis')
        axes[0, 1].set_title('Design Space Exploration')
        axes[0, 1].set_xlabel('Beam Height (m)')
        axes[0, 1].set_ylabel('Pier Spacing (m)')
        
        # 性能对比
        final_designs = self.population[:5]  # 展示前5个设计
        designs_data = []
        for i, design in enumerate(final_designs):
            weight, stress, deflection = design.calculate_performance()
            designs_data.append({
                'weight': weight,
                'stress': stress,
                'deflection': deflection,
                'fitness': design.fitness
            })
        
        # 归一化显示
        metrics = ['weight', 'stress', 'deflection', 'fitness']
        metric_names = ['Weight', 'Stress', 'Deflection', 'Fitness']
        x = np.arange(len(final_designs))
        width = 0.2
        
        for idx, metric in enumerate(metrics):
            values = [d[metric] for d in designs_data]
            # 对前三个指标取倒数以便于比较（值越小越好）
            if metric != 'fitness':
                values = [1/v if v != 0 else 0 for v in values]
            axes[1, 0].bar(x + idx*width, values, width, label=metric_names[idx])
        
        axes[1, 0].set_title('Performance Comparison of Top Designs')
        axes[1, 0].set_xlabel('Design ID')
        axes[1, 0].set_ylabel('Normalized Performance (1/value)')
        axes[1, 0].legend()
        
        # 最佳设计示意图
        best_design = max(self.population, key=lambda x: x.fitness)
        self.visualize_bridge_design(best_design, axes[1, 1])
        
        plt.tight_layout()
        plt.show()
    
    def visualize_bridge_design(self, design, ax):
        """绘制桥梁设计示意图"""
        beam_height, pier_spacing, num_piers, material_type = design.params
        material_names = ['Concrete', 'Steel', 'Composite']
        
        # 简化的桥梁绘制
        total_length = pier_spacing * (num_piers - 1)
        
        # 绘制桥面
        ax.plot([0, total_length], [beam_height, beam_height], 'k-', linewidth=3, label='Deck')
        
        # 绘制桥墩
        for i in range(num_piers):
            x = i * pier_spacing
            ax.plot([x, x], [0, beam_height], 'b-', linewidth=2, label='Pier' if i == 0 else "")
        
        ax.set_xlim(-5, total_length + 5)
        ax.set_ylim(0, beam_height * 2)
        ax.set_title(f'Best Design: {material_names[material_type]}\n'
                    f'Height: {beam_height:.1f}m, Spans: {pier_spacing:.1f}m\n'
                    f'Fitness: {design.fitness:.6f}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 显示性能指标
        weight, stress, deflection = design.calculate_performance()
        ax.text(0.02, 0.98, f'Weight: {weight/1000:.1f} ton\nStress: {stress:.1f} MPa\nDeflection: {deflection:.1f} mm', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 运行生成式设计
if __name__ == "__main__":
    span = 100  # 桥梁总跨度
    
    print("Starting Generative Bridge Design...")
    print("=" * 50)
    
    # 创建优化器并运行
    optimizer = GenerativeBridgeDesign(population_size=30, generations=50)
    optimizer.evolve()
    
    # 显示最终结果
    best_design = max(optimizer.population, key=lambda x: x.fitness)
    weight, stress, deflection = best_design.calculate_performance(span)
    
    print("\n" + "=" * 50)
    print("FINAL BEST DESIGN:")
    print(f"Beam Height: {best_design.params[0]:.2f} m")
    print(f"Pier Spacing: {best_design.params[1]:.2f} m")
    print(f"Number of Piers: {best_design.params[2]}")
    material_names = ['Concrete', 'Steel', 'Composite']
    print(f"Material: {material_names[best_design.params[3]]}")
    print(f"Total Weight: {weight/1000:.1f} tons")
    print(f"Max Stress: {stress:.1f} MPa")
    print(f"Max Deflection: {deflection:.1f} mm")
    print(f"Fitness: {best_design.fitness:.6f}")
    
    # 可视化结果
    optimizer.visualize_results()