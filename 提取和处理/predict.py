import torch
from model import PI_GNN
from torch_geometric.utils import subgraph
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
import numpy as np
import os

def predict_and_save():
    # 1. 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "model_safe_test.pth"     # 刚才训练保存的模型
    data_path = "vessel_graph_training.pt" # 数据文件
    output_vtp = "prediction_result.vtp"   # 输出文件
    
    print(f"🚀 开始推理...")
    
    # 2. 加载数据
    try:
        data = torch.load(data_path, weights_only=False)
    except:
        data = torch.load(data_path)
        
    # 获取元数据用于归一化
    global_centroid = data.centroid.to(device)
    global_scale = data.scale
    
    # 3. 加载模型
    print(f"📦 加载模型: {model_path}")
    model = PI_GNN(in_channels=9, hidden_channels=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # 切换到评估模式
    
    # 4. 推理 (分块推理以防爆显存)
    print("🏃 正在进行全量预测 (分块处理)...")
    
    num_nodes = data.num_nodes
    batch_size = 10000 # 推理时不需要算梯度，Batch 可以大一点
    
    # 结果容器
    all_u_pred = []
    all_p_pred = []
    
    with torch.no_grad(): # 关闭梯度计算
        for i in range(0, num_nodes, batch_size):
            end_idx = min(i + batch_size, num_nodes)
            node_indices = torch.arange(i, end_idx)
            
            # 构建子图
            edge_index, _ = subgraph(node_indices, data.edge_index, relabel_nodes=True)
            
            # 准备输入
            x_static = data.x[node_indices, :6].to(device)
            pos = data.pos[node_indices].to(device)
            
            # 动态计算归一化坐标
            pos_norm = (pos - global_centroid) / (global_scale + 1e-8)
            x_in = torch.cat([x_static, pos_norm], dim=1)
            edge_index = edge_index.to(device)
            
            # 预测
            u, p = model(x_in, edge_index)
            
            all_u_pred.append(u.cpu().numpy())
            all_p_pred.append(p.cpu().numpy())
            
            print(f"   - 已预测 {end_idx}/{num_nodes} 个点")

    # 合并结果
    u_final = np.concatenate(all_u_pred, axis=0)
    p_final = np.concatenate(all_p_pred, axis=0)
    
    print("✅ 预测完成，正在保存为 VTP...")

    # 5. 保存为 VTP (供 Paraview 查看)
    # 创建点云
    points = vtk.vtkPoints()
    # data.pos 是 Tensor，转为 numpy
    pos_np = data.pos.numpy()
    for k in range(num_nodes):
        points.InsertNextPoint(pos_np[k])
        
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    
    # 添加预测结果 (Velocity)
    u_array = numpy_to_vtk(u_final, deep=1)
    u_array.SetName("Predicted_Velocity")
    polydata.GetPointData().AddArray(u_array)
    
    # 添加预测结果 (Pressure)
    p_array = numpy_to_vtk(p_final, deep=1)
    p_array.SetName("Predicted_Pressure")
    polydata.GetPointData().AddArray(p_array)
    
    # 添加真实标签 (如果存在) 用于对比
    if hasattr(data, 'y') and data.y is not None:
        y_np = data.y.numpy()
        u_true = numpy_to_vtk(y_np[:, 0:3], deep=1)
        u_true.SetName("True_Velocity")
        polydata.GetPointData().AddArray(u_true)
        
        p_true = numpy_to_vtk(y_np[:, 3:4], deep=1)
        p_true.SetName("True_Pressure")
        polydata.GetPointData().AddArray(p_true)
        
        # 计算误差
        error = np.abs(u_final - y_np[:, 0:3])
        err_array = numpy_to_vtk(error, deep=1)
        err_array.SetName("Velocity_Error")
        polydata.GetPointData().AddArray(err_array)

    # 写入文件
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_vtp)
    writer.SetInputData(polydata)
    writer.Write()
    
    print(f"💾 结果已保存: {output_vtp}")
    print("请使用 Paraview 打开该文件，对比 Predicted_Velocity 和 True_Velocity。")

if __name__ == "__main__":
    predict_and_save()