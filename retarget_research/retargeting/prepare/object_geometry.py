"""提供重定向准备与运行阶段共享的物体世界几何函数。

输入：数据集物体目录、实例缩放、旋转矩阵和地面间隙。
输出：与CPU PhysX初始化一致的世界坐标表面顶点。
内部逻辑：加载`decomposed.obj`，旋转缩放后沿Z平移到地面上方。
作用：统一专家接触分析、目标手接触分析和物体感知优化的几何口径。
"""

import numpy as np
import trimesh


def transformed_object_vertices(object_dir, scale, rotation, clearance=0.005):
    """取得按数据姿态摆到地面后的物体表面顶点。

    输入：物体目录、缩放、3×3旋转矩阵和离地间隙。
    输出：形状`(V,3)`的世界坐标顶点。
    逻辑：加载`decomposed.obj`，先旋转缩放，再平移使最低Z等于clearance。
    作用：让所有物体感知模块引用同一份确定性表面几何。
    """
    vertices, _ = transformed_object_surface(
        object_dir, scale, rotation, clearance
    )
    return vertices


def transformed_object_surface(object_dir, scale, rotation, clearance=0.005):
    """取得摆地后的物体顶点及对应世界表面法向。

    输入：物体目录、统一缩放、3×3旋转矩阵和离地间隙。
    输出：`(V,3)`世界顶点和同形状单位法向。
    内部逻辑：顶点旋转缩放平移；法向只旋转并重新单位化。
    作用：在保持物体初始化口径一致的同时支持接触方向和对向夹持分析。
    """
    mesh = trimesh.load_mesh(object_dir / "coacd" / "decomposed.obj", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    rotation = np.asarray(rotation, dtype=np.float32)
    vertices = vertices @ rotation.T * float(scale)
    vertices[:, 2] += float(clearance) - float(vertices[:, 2].min())
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32) @ rotation.T
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    return vertices, normals
