def matmul_3d(A, B):
    """
    Perkalian matriks untuk array 3 dimensi (batch matrix multiplication)
    A: List 3D dengan shape (batch, m, n)
    B: List 3D dengan shape (batch, n, p)
    hasil: List 3D dengan shape (batch, m, p)
    """

    batch_size = len(A)
    m = len(A[0])       # jumlah baris A
    n = len(A[0][0])    # jumlah kolom A / baris B
    p = len(B[0][0])    # jumlah kolom B

    # Validasi ukuran batch dan dimensi dalam
    if len(B) != batch_size:
        raise ValueError("Jumlah batch A dan B harus sama")

    if len(B[0]) != n:
        raise ValueError("Kolom A harus sama dengan baris B")

    hasil = []

    for b in range(batch_size):
        matrix_hasil = []

        for i in range(m):
            baris_hasil = []

            for j in range(p):
                total = 0

                for k in range(n):
                    total += A[b][i][k] * B[b][k][j]

                baris_hasil.append(total)

            matrix_hasil.append(baris_hasil)

        hasil.append(matrix_hasil)

    return hasil


# Contoh penggunaan
A = [
    [[1, 2], [3, 4]],    # matrix 1 (2x2)
    [[5, 6], [7, 8]]     # matrix 2 (2x2)
]

B = [
    [[2, 0], [1, 3]],    # matrix 1 (2x2)
    [[1, 1], [0, 2]]     # matrix 2 (2x2)
]


hasil = matmul_3d(A, B)

for idx, m in enumerate(hasil):
    print(f"Hasil batch ke-{idx}:")
    for baris in m:
        print(baris)
    print()