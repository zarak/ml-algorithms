import week2

N_values = [500, 1000, 1500, 2000]

def question3():
    return [week2.hoeffding_RHS(0.05, N, 100) for N in N_values]
