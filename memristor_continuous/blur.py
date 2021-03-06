import numpy as np

blur3x3 = np.array([
                    [[1/9, 1/9, 1/9],
                     [1/9, 1/9, 1/9],
                     [1/9, 1/9, 1/9]]
                    ,
                    [[1/13, 1/13, 1/13],
                     [1/13, 5/13, 1/13],
                     [1/13, 1/13, 1/13]]
                    ,
                    [[1/16, 1/16, 1/16],
                     [1/16, 8/16, 1/16],
                     [1/16, 1/16, 1/16]]
                    ,
                    [[1/24, 3/24, 3/24],
                     [3/24, 8/24, 3/24],
                     [1/24, 1/24, 3/24]]])

blur5x5 = np.array([
                    [[1/25, 1/25, 1/25, 1/25, 1/25],
                     [1/25, 1/25, 1/25, 1/25, 1/25],
                     [1/25, 1/25, 1/25, 1/25, 1/25],
                     [1/25, 1/25, 1/25, 1/25, 1/25],
                     [1/25, 1/25, 1/25, 1/25, 1/25]]
                    ,
                    [[1/39, 1/39, 1/39,  1/39, 1/39],
                     [1/39, 1/39, 1/39,  1/39, 1/39],
                     [1/39, 1/39, 15/39, 1/39, 1/39],
                     [1/39, 1/39, 1/39,  1/39, 1/39],
                     [1/39, 1/39, 1/39,  1/39, 1/39]]
                    ,
                    [[1/48, 1/48, 1/48,  1/48, 1/48],
                     [1/48, 1/48, 1/48,  1/48, 1/48],
                     [1/48, 1/48, 24/48, 1/48, 1/48],
                     [1/48, 1/48, 1/48,  1/48, 1/48],
                     [1/48, 1/48, 1/48,  1/48, 1/48]]
                    ,
                    [[1/64, 1/64, 2/64,  1/64, 1/64],
                     [1/64, 2/64, 3/64,  2/64, 1/64],
                     [2/64, 3/64, 24/64, 3/64, 2/64],
                     [1/64, 2/64, 3/64,  2/64, 1/64],
                     [1/64, 1/64, 2/64,  1/64, 1/64]]])

