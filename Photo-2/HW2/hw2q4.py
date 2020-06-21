import numpy as np

if __name__ == '__main__':
    # with open('pts.txt', 'r', encoding='UTF-8') as file:
    #     lines = file.readlines()
    #     data = []
    #     for line in lines:
    #         line = line.split()
    #         data.append(np.array([line[0], line[1], line[2]]).astype(float))
    #
    # data = np.array(data)
    #
    # # LS Adjustment
    # A = np.hstack((data[:, 0:2], np.ones((data.shape[0], 1))))
    # l = data[:, 2]
    #
    # x_ls = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, l))
    # x_ls = np.array([x_ls[0], x_ls[1], -1, x_ls[2]])
    #
    # # Homogeneous
    # center = np.array([np.average(data[:, 0]), np.average(data[:, 1]), np.average(data[:, 2])])
    # t_data = data - center  # translating to center
    # p_A = t_data
    #
    # egi_vals, egi_vect = np.linalg.eig(np.dot(p_A.T, p_A))
    # min_egi_val_index = np.argmin(egi_vals)
    # x_p = egi_vect[:, min_egi_val_index]
    #
    # d_p = -np.average(np.dot(data, x_p))
    # x_p = np.hstack((x_p,d_p))
    #
    # diff_ls = np.linalg.norm(np.dot(data, x_ls[0:3]) + x_ls[-1])
    # diff_p = np.linalg.norm(np.dot(data, x_p[0:3]) + x_p[-1])

    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print('hi')
