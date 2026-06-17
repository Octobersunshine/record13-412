import numpy as np


def solve_linear_system(A, b):
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if A.ndim != 2:
        raise ValueError(f"系数矩阵必须是二维数组，当前维度: {A.ndim}")
    if b.ndim != 1:
        raise ValueError(f"常数向量必须是一维数组，当前维度: {b.ndim}")
    n, m = A.shape
    if n != m:
        raise ValueError(f"系数矩阵必须是方阵，当前形状: {A.shape}")
    if b.shape[0] != n:
        raise ValueError(f"常数向量长度 {b.shape[0]} 与矩阵维度 {n} 不匹配")

    augmented = np.column_stack((A, b))
    rank_A = np.linalg.matrix_rank(A)
    rank_aug = np.linalg.matrix_rank(augmented)

    if rank_A != rank_aug:
        return "无解", None

    if rank_A < n:
        x_particular = np.linalg.pinv(A) @ b
        return "无穷多解", x_particular

    x = np.linalg.solve(A, b)
    return "唯一解", x


def main():
    print("=== 线性方程组求解服务 ===")
    print("格式: Ax = b，输入系数矩阵 A 和常数向量 b\n")

    A_input = input("请输入系数矩阵 A（每行用分号分隔，元素用空格分隔）: ")
    b_input = input("请输入常数向量 b（元素用空格分隔）: ")

    try:
        rows = A_input.strip().split(";")
        A = [[float(x) for x in row.strip().split()] for row in rows]
        b = [float(x) for x in b_input.strip().split()]

        status, solution = solve_linear_system(A, b)

        print("\n--- 求解结果 ---")
        print(f"系数矩阵 A:\n{np.array(A)}")
        print(f"常数向量 b: {np.array(b)}")
        print(f"解的状态: {status}")
        if solution is not None:
            print(f"解 x: {solution}")
            print("\n验证 Ax:")
            print(np.array(A) @ solution)
    except ValueError as e:
        print(f"输入错误: {e}")


if __name__ == "__main__":
    main()
