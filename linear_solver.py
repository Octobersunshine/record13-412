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


def solve_least_squares(A, b):
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if A.ndim != 2:
        raise ValueError(f"系数矩阵必须是二维数组，当前维度: {A.ndim}")
    if b.ndim != 1:
        raise ValueError(f"常数向量必须是一维数组，当前维度: {b.ndim}")
    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError(f"常数向量长度 {b.shape[0]} 与矩阵行数 {m} 不匹配")

    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

    if rank == n and m >= n:
        if m == n:
            residual_norm = np.linalg.norm(A @ x - b)
            if np.isclose(residual_norm, 0.0):
                return "唯一解", x, 0.0
            else:
                return "最小二乘解", x, residual_norm
        else:
            residual_norm = residuals[0] if len(residuals) > 0 else np.linalg.norm(A @ x - b)
            return "最小二乘解", x, float(residual_norm)
    else:
        x_particular = np.linalg.pinv(A) @ b
        residual_norm = np.linalg.norm(A @ x_particular - b)
        return "最小二乘解（秩亏）", x_particular, residual_norm


def main():
    print("=== 线性方程组求解服务 ===")
    print("1. 方阵方程组求解（Ax = b）")
    print("2. 超定方程组最小二乘解（Ax ≈ b）\n")

    choice = input("请选择求解模式 (1/2): ").strip()

    A_input = input("请输入系数矩阵 A（每行用分号分隔，元素用空格分隔）: ")
    b_input = input("请输入常数向量 b（元素用空格分隔）: ")

    try:
        rows = A_input.strip().split(";")
        A = [[float(x) for x in row.strip().split()] for row in rows]
        b = [float(x) for x in b_input.strip().split()]

        print("\n--- 求解结果 ---")
        print(f"系数矩阵 A:\n{np.array(A)}")
        print(f"常数向量 b: {np.array(b)}")

        if choice == "2":
            status, solution, residual = solve_least_squares(A, b)
            print(f"解的状态: {status}")
            print(f"最小二乘解 x: {solution}")
            print(f"残差 ||Ax - b||: {residual}")
            print("\n验证 Ax:")
            print(np.array(A) @ solution)
        else:
            status, solution = solve_linear_system(A, b)
            print(f"解的状态: {status}")
            if solution is not None:
                print(f"解 x: {solution}")
                print("\n验证 Ax:")
                print(np.array(A) @ solution)
    except ValueError as e:
        print(f"输入错误: {e}")


if __name__ == "__main__":
    main()
